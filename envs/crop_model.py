from configparser import ConfigParser
from collections import namedtuple
import datetime as dt
import os

import gym
import numpy as np
import pandas as pd
import requests

from utils import verify_output_path


class CropParam:
    def __init__(self, filepath=None):
        self.PHU = None  # potential heat units required for maturity
        self.BE = None  # crop parameter: energy to biomass
        self.HI = None  # potential harvest index
        self.To = None  # optimal temperature
        self.Tb = None  # base temperature
        self.LAI_max = None  # max leaf area index potential
        self.HUI0 = None  # heat unit index value when leaf area index starts declining
        self.ah1 = None  # crop parameters that determine the shape of the leaf-area-index development curve
        self.ah2 = None  # crop parameters that determine the shape of the leaf-area-index development curve
        self.af1 = None  # crop parameters for frost sensitivity
        self.af2 = None  # crop parameters for frost sensitivity
        self.ad = None  # crop parameter that governs leaf area index decline rate
        self.ALT = None  # aluminum tolerance index number
        self.CAF = None  # critical aeration factor
        self.HMX = None  # maximum crop height
        self.RDMX = None  # maximum root depth
        self.WSYF = None  # water stress factor for adjusting harvest index
        self.bn1 = None  # crop parameters for plant N concentration equation
        self.bn2 = None  # crop parameters for plant N concentration equation
        self.bn3 = None  # crop parameters for plant N concentration equation
        self.bp1 = None  # crop parameters for plant P concentration equation
        self.bp2 = None  # crop parameters for plant P concentration equation
        self.bp3 = None  # crop parameters for plant P concentration equation
        self.E0 = None  # evaporation per day

        if filepath is not None:
            self.load_from_file(filepath)

    def load_from_file(self, filepath):
        parser = ConfigParser()
        parser.optionxform = str
        parser.read(filepath)
        for key, item in parser['DEFAULT'].items():
            setattr(self, key, float(item))


class LocParam:
    def __init__(self, filepath=None):
        self.lat = None  # latitude
        self.lon = None  # longitude
        self.HRLT_min = None  # min day length
        self.SW = None  # soil water level
        self.c_LP = None  # labile phosphorus concentration in the soil
        self.PO = None  # soil porosity

        if filepath is not None:
            self.load_from_file(filepath)

    def load_from_file(self, filepath):
        parser = ConfigParser()
        parser.optionxform = str
        parser.read(filepath)
        for key, item in parser['DEFAULT'].items():
            setattr(self, key, float(item))


class WeatherData:
    POWER_VARIABLES = ['T2M_MIN', 'T2M_MAX', 'T2M', 'PRECTOT', 'ALLSKY_SFC_SW_DWN']
    MJ_M2_TO_LANG = lambda x: x / 41868

    def __init__(self, lat, lon, cache_dir='./envs/weather_data/'):
        self.lat = lat
        self.lon = lon
        self.cache_dir = cache_dir

        self.df: pd.DataFrame = pd.DataFrame()

        if not self.read_cache():
            data = self.query_nasa_power()
            self.process_nasa_power_data(data)
            self.save_cache()

    def cache_file_path(self):
        filename = f'lat_{self.lat:.1f}_lon_{self.lon:.1f}.pkl'
        return os.path.join(self.cache_dir, filename)

    def read_cache(self):
        filepath = self.cache_file_path()
        try:
            self.df = pd.read_pickle(filepath)
            return True
        except:
            return False

    def save_cache(self):
        filepath = self.cache_file_path()
        verify_output_path(filepath)
        pd.to_pickle(self.df, filepath)

    def query_nasa_power(self):
        start_date = dt.date(1984, 1, 1)
        end_date = dt.date(2020, 1, 1)

        # build URL for retrieving data
        server = 'https://power.larc.nasa.gov/cgi-bin/v1/DataAccess.py'
        payload = {'request': 'execute',
                   'identifier': 'SinglePoint',
                   'parameters': ','.join(self.POWER_VARIABLES),
                   'lat': self.lat,
                   'lon': self.lon,
                   'startDate': start_date.strftime('%Y%m%d'),
                   'endDate': end_date.strftime('%Y%m%d'),
                   'userCommunity': 'AG',
                   'tempAverage': 'DAILY',
                   'outputList': 'JSON',
                   'user': 'anonymous'
                   }
        req = requests.get(server, params=payload)

        if req.status_code != 200:
            raise RuntimeError('Failed to retrieve weather data from NASA POWER: {req.status_code} from {req.url}')

        result = req.json()
        if not result:
            raise RuntimeError('Failed to retrieve weather data from NASA POWER')

        return result

    def process_nasa_power_data(self, data):
        fill_value = float(data['header']['fillValue'])

        d = {}
        for var in self.POWER_VARIABLES:
            s = pd.Series(data['features'][0]['properties']['parameter'][var])
            s[s == fill_value] = np.NaN
            d[var] = s
        self.df = pd.DataFrame(d)

        # adjust unit
        self.df['ALLSKY_SFC_SW_DWN'] = self.df['ALLSKY_SFC_SW_DWN'].apply(WeatherData.MJ_M2_TO_LANG)

        # remove rows that have missing data
        self.df = self.df.dropna(axis='index')

    def get_weather(self, date: dt.date) -> (float, float, float, float, float):
        return self.df.loc[date.strftime('%Y%m%d')].values


class CropEnv(gym.Env):
    STATE_KEYS = ('date', 'day', 't_min', 't_max', 't_avg', 'rain', 'ra',
                  'CHT', 'HU_total', 'HUI', 'LAI', 'LAI0', 'B',
                  'UN_total', 'UP_total')

    State = namedtuple('State', STATE_KEYS)

    def __init__(self, crop_config='./envs/crops/corn.ini', loc_config='./envs/locations/davis.ini',
                 max_iter=365):
        # day, min temperature, max temperature, avg temperature, sunlight radiation, amount of rain
        # plant height, leaf area index
        self.observation_space = gym.spaces.discrete.Discrete(8)
        # water, nitrogen, phosphorus, harvest
        # self.action_space = gym.spaces.discrete.Discrete(4)
        # water, nitrogen, phosphorus
        self.action_space = gym.spaces.discrete.Discrete(3)

        self.max_iter = max_iter

        self.crop = CropParam(crop_config)
        self.loc = LocParam(loc_config)
        self.weather_data = WeatherData(self.loc.lat, self.loc.lon)

        self.start_date: dt.date = dt.date.today()
        self.state: CropEnv.State = CropEnv.State(*([None] * 15))

        self.reset()

    def reset(self):
        # start from a random date from 1960 to 1982
        start_year = np.random.randint(1984, 2000)
        num_days = (dt.date(start_year + 1, 1, 1) - dt.date(start_year, 1, 1)).days
        days = np.random.randint(num_days)
        self.start_date = dt.date(start_year, 1, 1) + dt.timedelta(days=days)

        # get weather data
        t_min, t_max, t_avg, rain, ra = self.weather_data.get_weather(self.start_date)

        # reset state
        self.state = CropEnv.State(
            date=self.start_date,  # actual date
            day=(self.start_date - dt.date(self.start_date.year, 1, 1)).days,  # day number
            t_min=t_min, t_max=t_max, t_avg=t_avg, rain=rain, ra=ra,
            HU_total=0,  # total heat unit
            HUI=0,  # heat unit index
            LAI=0,  # leaf area index
            LAI0=-1,  # LAI when the leaf decline starts, or -1 before that
            CHT=0,  # crop height
            B=0,  # crop biomass
            UN_total=0,  # total amount of nitrogen applied
            UP_total=0,  # total amount of phosphorus applied
        )

        return np.array([self.state.day, t_min, t_max, t_avg, rain, ra, self.state.CHT, self.state.LAI])

    def step(self, action):
        # irrigation, nitrogen, phosphorus, harvest = action.squeeze()
        irrigation, nitrogen, phosphorus = action.squeeze()

        # update date and weather data
        date = self.state.date + dt.timedelta(days=1)
        day = date.day
        t_min, t_max, t_avg, rain, ra = self.weather_data.get_weather(date)

        water = irrigation + rain

        # update heat unit
        HU = self.heat_unit(self.crop, t_min, t_max)
        HU_total = self.state.HU_total + HU

        # update heat unit index
        HUI = self.heat_unit_index(self.crop, HU_total)

        # check if leaf declination starts
        LAI0 = self.state.LAI if HUI >= self.crop.HUI0 else -1

        # update nitrogen and phosphorus
        UN = self.nitrogen_uptake(nitrogen, self.loc.SW)
        UN_total = self.state.UN_total + UN
        UP = self.phosphorus_uptake(self.crop, phosphorus, self.state.B, self.state.UP_total, HUI, self.loc.c_LP)
        UP_total = self.state.UP_total + UP

        # compute the 5 stresses
        WS = self.water_stress(self.crop, water, self.state.LAI)
        NS = self.nitrogen_stress(self.crop, self.state.B, UN_total, HUI)
        PS = self.phosphorus_stress(self.crop, self.state.B, UP_total, HUI)
        TS = self.temperature_stress(self.crop, t_avg)
        AS = self.aeration_stress(self.crop, water, self.loc.PO)

        # compute crop growth constraint
        REG = min(WS, NS, PS, TS, AS)

        # update leaf area index
        LAI = self.leaf_area_index(self.crop, self.state.LAI, LAI0, REG, HUI, self.state.HUI)

        # calculate frost damage
        delta_B_FRST = self.frost_damage(self.crop, date, self.loc.lat, self.loc.HRLT_min, t_min, self.state.B, HUI)

        # update biomass
        delta_Bp = self.potential_biomass_growth(self.crop, date, ra, LAI, self.loc.lat)
        delta_B = REG * delta_Bp
        B = self.state.B + delta_B + delta_B_FRST

        # update crop height
        CHT = self.height(self.crop, HUI)

        # update state
        self.state = CropEnv.State(
            date=date,  # actual date
            day=(date - dt.date(date.year, 1, 1)).days,  # day number in year
            t_min=t_min, t_max=t_max, t_avg=t_avg, rain=rain, ra=ra,
            HU_total=HU_total,  # total heat unit
            HUI=HUI,  # heat unit index
            LAI=LAI,  # leaf area index
            LAI0=LAI0,  # LAI when the leaf decline starts, or -1 before that
            CHT=CHT,  # crop height
            B=B,  # crop biomass
            UN_total=UN_total,  # total amount of nitrogen applied
            UP_total=UP_total,  # total amount of phosphorus applied
        )

        harvest = HUI >= 0.99

        observation = np.array([day, t_min, t_max, t_avg, rain, ra, CHT, LAI])
        reward = self.total_yield(self.crop, HUI, B) if harvest else 0
        done = True if harvest or (date - self.start_date).days > self.max_iter else False

        return observation, reward, done, None

    @staticmethod
    def heat_unit(crop: CropParam, T_min, T_max):
        # heat unit
        HU = (T_min + T_max) / 2 - crop.Tb
        HU = max(HU, 0)
        return HU

    @staticmethod
    def heat_unit_index(crop: CropParam, HU_total):
        # heat unit index
        HUI = HU_total / crop.PHU
        HUI = np.clip(HUI, 0, 1)
        return HUI

    @staticmethod
    def water_stress(crop: CropParam, u, LAI):
        # potential water use
        Ep = crop.E0 * LAI / 3
        Ep = min(Ep, crop.E0)
        # water stress
        if LAI > 0:
            WS = u / Ep
        else:
            WS = 1
        return WS

    @staticmethod
    def nitrogen_stress(crop: CropParam, B, UN_total, HUI):
        # optimal nitrogen concentration
        c_NB = crop.bn1 + crop.bn2 * np.exp(-crop.bn3 * HUI)
        if B > 0:
            # scaling factor for the nitrogen stress factor
            SN_S = 2 * (1 - UN_total / (c_NB * B))
            SN_S = np.clip(SN_S, -10, 10)
            # nitrogen stress factor
            SN = 1 - (SN_S / (SN_S + np.exp(3.39 - 10.93 * SN_S)))
        else:
            SN = 1
        return SN

    @staticmethod
    def nitrogen_uptake(WNO3, SW):
        # nitrogen uptake
        UN = WNO3 / SW
        return UN

    @staticmethod
    def phosphorus_stress(crop: CropParam, B, UP_total, HUI):
        # optimal phosphorus concentration
        c_PB = crop.bp1 + crop.bp2 * np.exp(-crop.bp3 * HUI)
        if B > 0:
            # scaling factor for the phosphorus stress factor
            SP_S = 2 * (1 - UP_total / (c_PB * B))
            SP_S = np.clip(SP_S, -10, 10)
            # phosphorus stress factor
            PN = 1 - (SP_S / (SP_S + np.exp(3.39 - 10.93 * SP_S)))
        else:
            PN = 1
        return PN

    @staticmethod
    def phosphorus_uptake(crop: CropParam, P, B, UP_total, HUI, c_LP):
        # optimal phosphorus concentration
        c_PB = crop.bp1 + crop.bp2 * np.exp(-crop.bp3 * HUI)
        # phosphorus demand
        UPD = c_PB * B - UP_total
        # labile phosphorus factor for uptake
        LF = 0.1 + 0.9 * (c_LP / (c_LP + 117 * np.exp(-0.283 * c_LP)))
        # phosphorus supplied by the soil
        UPS = 1.5 * UPD * LF
        # phosphorus uptake
        UP = UPS + P
        return UP

    @staticmethod
    def temperature_stress(crop: CropParam, T_avg):
        # temperature stress
        TS = np.sin(np.pi / 2 * ((T_avg - crop.Tb) / (crop.To - crop.Tb)))
        TS = np.clip(TS, 0, 1)
        return TS

    @staticmethod
    def aeration_stress(crop: CropParam, u, PO):
        # saturation factor
        SAT = u / PO - crop.CAF
        # aeration stress
        if SAT > 0:
            AS = 1 - SAT / (SAT + np.exp(-1.291 - 56.1 * SAT))
        else:
            AS = 1

        return AS

    @staticmethod
    def leaf_area_index(crop: CropParam, prev_LAI, LAI0, REG, HUI, prev_HUI):
        # from the emergence to the start of leaf decline
        if LAI0 == -1:
            # change in heat unit factor
            delta_HUF = CropEnv.heat_unit_factor(crop, HUI) \
                        - CropEnv.heat_unit_factor(crop, prev_HUI)
            # change in leaf area index
            delta_LAI = delta_HUF \
                        * crop.LAI_max \
                        * (1 - np.exp(5 * prev_LAI - crop.LAI_max)) \
                        * np.sqrt(REG)
            LAI = prev_LAI + delta_LAI
        # from the start of leaf decline to the end of the growing season
        else:
            LAI = LAI0 * (1 - HUI / 1 - crop.HUI0) ** crop.ad
        return LAI

    @staticmethod
    def heat_unit_factor(crop: CropParam, HUI):
        # heat unit factor
        HUF = HUI / (HUI + np.exp(crop.ah1 - crop.ah2 * HUI))
        return HUF

    @staticmethod
    def potential_biomass_growth(crop: CropParam, date: dt.date, RA, LAI, LAT):
        # intercepted photosynthetic active radiation
        PAR = 0.5 * RA * (1 - np.exp(-0.65 * LAI))
        # change in day length
        delta_HRLT = CropEnv.day_length(date, LAT) \
                     - CropEnv.day_length(date - dt.timedelta(days=1), LAT)
        # daily potential increase in biomass
        delta_Bp = 0.001 * crop.BE * PAR * (1 + delta_HRLT) ** 3
        return delta_Bp

    @staticmethod
    def frost_damage(crop: CropParam, date: dt.date, LAT, HRLT_min, T_min, B, HUI):
        # day length
        HRLT = CropEnv.day_length(date, LAT)
        # day length reduction factor
        FHR = 0.35 * (1 - HRLT / (HRLT_min + 1))
        # frost reduction factor
        if T_min < 1:
            FRST = -T_min / (-T_min - np.exp(crop.af1 + crop.af2 * T_min))
        else:
            FRST = 0
        # frost damage in biomass
        delta_B_FRST = 0.5 * B * (1 - HUI) * max(FHR, FRST)
        return delta_B_FRST

    @staticmethod
    def day_length(date: dt.date, LAT):
        # sun declination angle
        SD = 0.4102 * np.sin(2 * np.pi / 365 * (date.day - 80.25))
        # day length
        HRLT = 7.64 * np.arccos(
            -np.sin(2 * np.pi * LAT / 365 * np.sin(SD) - 0.044)
            / np.cos(2 * np.pi * LAT / 365) * np.cos(SD)
        )
        return HRLT

    @staticmethod
    def height(crop: CropParam, HUI):
        # crop height
        CHT = crop.HMX * np.sqrt(CropEnv.heat_unit_factor(crop, HUI))
        return CHT

    @staticmethod
    def total_yield(crop: CropParam, HUI, B):
        # heat unit factor that affects harvest index
        HUFH = HUI / (HUI + np.exp(6.5 - 10 * HUI))
        # actual harvest index
        HIA = crop.HI * HUFH
        # yield
        YLD = B * HIA
        return YLD

    def render(self, mode='human'):
        pass

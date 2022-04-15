# %%
import numpy as np
import pandas as pd
import pickle
import os
import zipfile
import warnings

warnings.filterwarnings("ignore")
os.chdir("/Users/mohammadanas/Desktop/anas github/IDS-705-Final-Project/")
# print(os.chdir(os.getenv('path_env')))

# %%
# Loading shortlisted feature from pickle file
with open("./trained_models/imp_features.pkl", "rb") as f:
    features = pickle.load(f)

shortlisted_features = list(features.keys())[:150]

# %%
# Loading train, test and validation files

train = pd.read_csv("./data/train.csv")
train.name = "Train"
val = pd.read_csv("./data/val.csv")
val.name = "Validation"
test = pd.read_csv("./data/test.csv")
test.name = "Test"

# %%
cols_to_drop = [
    "IsBeta",
    "AutoSampleOptIn",
    "SMode",
    "Census_IsPortableOperatingSystem",
    "OrganizationIdentifier",
    "Census_InternalBatteryNumberOfCharges",
]


categorical_cols = [
    "IsSxsPassiveMode",
    "RtpStateBitfield",
    "AVProductStatesIdentifier",
    "AVProductsInstalled",
    "AVProductsEnabled",
    "HasTpm",  # think of dropping it
    "CountryIdentifier",
    "CityIdentifier",
    # "OrganizationIdentifier",
    "GeoNameIdentifier",
    "LocaleEnglishNameIdentifier",
    "Platform",
    "Processor",
    "OsVer",  # Think of dropping it
    "OsBuild",
    "OsSuite",
    "IsProtected",
    "IeVerIdentifier",
    "Firewall",
    "UacLuaenable",  # THINK of dropping
    "Census_OEMNameIdentifier",
    "Census_OEMModelIdentifier",
    "Census_ProcessorManufacturerIdentifier",
    "Census_ProcessorModelIdentifier",
    "Census_HasOpticalDiskDrive",
    "Census_PowerPlatformRoleName",
    "Census_OSVersion",
    "Census_OSArchitecture",
    "Census_OSBranch",  # OS version
    "Census_OSBuildNumber",  # OS version
    "Census_OSBuildRevision",  # OS version
    "Census_OSInstallLanguageIdentifier",  # think of dropping it
    "Census_OSUILocaleIdentifier",
    "Census_IsFlightsDisabled",
    "Census_FlightRing",
    "Census_FirmwareManufacturerIdentifier",
    "Census_FirmwareVersionIdentifier",
    "Census_IsSecureBootEnabled",
    "Census_IsVirtualDevice",
    "Census_IsTouchEnabled",
    "Census_IsPenCapable",
    "Census_IsAlwaysOnAlwaysConnectedCapable",
    "Wdft_IsGamer",
    "Wdft_RegionIdentifier",
    # "HasDetections"
]

nulls_cols = [
    "MachineIdentifier",
    "ProductName",
    "EngineVersion",
    "AppVersion",
    "AvSigVersion",
    "IsBeta",
    "RtpStateBitfield",
    "IsSxsPassiveMode",
    "AVProductStatesIdentifier",
    "AVProductsInstalled",
    "AVProductsEnabled",
    "HasTpm",
    "CountryIdentifier",
    "CityIdentifier",
    "OrganizationIdentifier",
    "GeoNameIdentifier",
    "LocaleEnglishNameIdentifier",
    "Platform",
    "Processor",
    "OsVer",
    "OsBuild",
    "OsSuite",
    "OsPlatformSubRelease",
    "OsBuildLab",
    "SkuEdition",
    "IsProtected",
    "AutoSampleOptIn",
    "SMode",
    "IeVerIdentifier",
    "SmartScreen",
    "Firewall",
    "UacLuaenable",
    "Census_MDC2FormFactor",
    "Census_DeviceFamily",
    "Census_OEMNameIdentifier",
    "Census_OEMModelIdentifier",
    "Census_ProcessorCoreCount",
    "Census_ProcessorManufacturerIdentifier",
    "Census_ProcessorModelIdentifier",
    "Census_PrimaryDiskTotalCapacity",
    "Census_PrimaryDiskTypeName",
    "Census_SystemVolumeTotalCapacity",
    "Census_HasOpticalDiskDrive",
    "Census_TotalPhysicalRAM",
    "Census_ChassisTypeName",
    "Census_InternalPrimaryDiagonalDisplaySizeInInches",
    "Census_InternalPrimaryDisplayResolutionHorizontal",
    "Census_InternalPrimaryDisplayResolutionVertical",
    "Census_PowerPlatformRoleName",
    "Census_InternalBatteryNumberOfCharges",
    "Census_OSVersion",
    "Census_OSArchitecture",
    "Census_OSBranch",
    "Census_OSBuildNumber",
    "Census_OSBuildRevision",
    "Census_OSEdition",
    "Census_OSSkuName",
    "Census_OSInstallTypeName",
    "Census_OSInstallLanguageIdentifier",
    "Census_OSUILocaleIdentifier",
    "Census_OSWUAutoUpdateOptionsName",
    "Census_IsPortableOperatingSystem",
    "Census_GenuineStateName",
    "Census_ActivationChannel",
    "Census_IsFlightsDisabled",
    "Census_FlightRing",
    "Census_FirmwareManufacturerIdentifier",
    "Census_FirmwareVersionIdentifier",
    "Census_IsSecureBootEnabled",
    "Census_IsVirtualDevice",
    "Census_IsTouchEnabled",
    "Census_IsPenCapable",
    "Census_IsAlwaysOnAlwaysConnectedCapable",
    "Wdft_IsGamer",
    "Wdft_RegionIdentifier",
    #'HasDetections'
]

# Features to drop post correlation checks
corr_cols_drop = [
    "Census_ProcessorCoreCount",
    "Census_PrimaryDiskTotalCapacity",
    "Census_SystemVolumeTotalCapacity",
    "Census_TotalPhysicalRAM",
    "Census_InternalPrimaryDiagonalDisplaySizeInInches",
    "Census_InternalPrimaryDisplayResolutionHorizontal",
    "Census_InternalPrimaryDisplayResolutionVertical",
    "AVProductsInstalled",
]

log_corr_cols_to_drop = [
    "log_Census_PrimaryDiskTotalCapacity",
    "log_Census_InternalPrimaryDisplayResolutionVertical",
]

# Categorical columns similar to other columns
# redundant information. Hence dropped.
cats_to_drop = [
    "MachineIdentifier",
    "OsBuildLab",
    "CityIdentifier",
    "GeoNameIdentifier",
    "LocaleEnglishNameIdentifier",
    "Census_ChassisTypeName",
    "Census_OSVersion",
    "Census_OSBranch",
    "Census_OSBuildNumber",
    "Census_OSSkuName",
    "Wdft_RegionIdentifier",
]


# %%
def drop_cols(df, *args):
    """Drop redundant columns."""
    for x in args:
        df.drop(columns=x, inplace=True)


def replace_nulls(df):
    """Replace nulls acc to data types."""
    for col in nulls_cols:
        if df[col].dtype.name in ["int64", "float64"]:
            df[col] = df[col].fillna(-999999999).copy()
        else:
            df[col].fillna("Missing", inplace=True)
    return df


def convert_num_to_cat(df):
    """Correct some numerical attributes to categorical."""
    for col in categorical_cols:
        df[col] = df[col].astype("str").copy()
    return df


def median_impute(df):
    """Function imputes the float attributes with the median."""
    for col in df.columns:
        if df[col].dtype.name in ["int64", "float64"]:
            df[col].replace(-999999999, df[col].median(), inplace=True)
    return df


def treat_outlier(df):
    """Remove the outliers based on 3 standard deviations away from mean."""
    for col in df.select_dtypes(include=np.number).columns.tolist():

        a = np.mean(df[col])
        b = np.std(df[col])

        df[col] = np.where(
            df[col] > a + 3 * b,
            a + 3 * b,
            np.where(df[col] < a - 3 * b, a - 3 * b, df[col]),
        )
    return df


def logtransform_num(df):
    """Taking the log of the numerical attributes."""
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    for col in num_cols:
        df["log_" + col] = np.log(df[col])
        # Doing the following for values 0 and very large values to avoid negative infs
        df["log_" + col] = (
            df["log_" + col]
            .replace(-np.inf, np.mean(df["log_" + col]) - 3 * np.std(df["log_" + col]))
            .copy()
        )
        df["log_" + col] = (
            df["log_" + col]
            .replace(np.nan, np.mean(df["log_" + col]) + 3 * np.std(df["log_" + col]))
            .copy()
        )
    return df


def convert_cat_to_num(df):
    """Convert attributes which were tagged as categorical but are numeric."""
    convert_to_num = ["AVProductsInstalled", "AVProductsEnabled"]

    for col in convert_to_num:
        df[col] = df[col].astype(float).astype(int).copy()

    return df


# df.drop(columns = corr_cols_drop, inplace = True)
# df.drop(columns= cats_to_drop, inplace=True)


def treat_firewall(df):
    """Impute missing values for firewall."""
    # NA means that firewall was not there
    # hence replaced by zero
    df["Firewall"].replace("-999999999.0", "0.0", inplace=True)
    return df


# %%
def pre_process_df(df1):
    """Function to implement all pre-processing steps in
    the order carried out in the train dataset."""
    df = df1.copy()
    df = replace_nulls(df)
    drop_cols(df, cols_to_drop)
    df = convert_num_to_cat(df)
    df = median_impute(df)
    df = treat_outlier(df)
    df = logtransform_num(df)
    df = convert_cat_to_num(df)
    drop_cols(df, corr_cols_drop, cats_to_drop, log_corr_cols_to_drop)
    treat_firewall(df)
    return df


# %%
class preprocessing_freq_encode:
    """Class will keep the most common occuring features
    based on the limit given."""

    def __init__(self, limit):
        """Iniatialize Class."""
        self.limit = limit
        self.dict = {}
        self.convert_num = []

    def fit(self, data_frame):
        """Perform preprocessing steps and Feature Encoding."""
        main_df = pre_process_df(data_frame)
        for i in list(main_df.select_dtypes(include="object").columns):
            if len(main_df[i].unique()) <= 2:
                self.convert_num.append(i)
            else:
                bool_ = main_df[i].value_counts(ascending=False) > self.limit
                vals = main_df[i].value_counts(ascending=False)[bool_]
                self.dict[i] = list(vals.index)

    def transform(self, data_frame):
        """Transform the Data prior Modelling."""
        main_df = pre_process_df(data_frame)
        data_frame1 = main_df.copy()
        for i in self.convert_num:
            data_frame1[i] = data_frame1[i].astype("float")
        for j in self.dict:
            for k in self.dict[j]:
                data_frame1[j + k] = np.where(data_frame1[j] == k, 1, 0)
        data_frame1.drop(columns=list(self.dict.keys()), inplace=True)
        data_frame_final = data_frame1[shortlisted_features].copy()
        return data_frame_final

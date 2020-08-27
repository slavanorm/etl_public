import warnings

warnings.simplefilter(action="ignore",)

from functools import wraps
import json
import time
import functools
from datetime import datetime, timedelta
import sys
import requests
import gspread as g
from gspread.utils import rowcol_to_a1
import pickle
import numpy as np
from pandas import json_normalize
import gspread_formatting as gsf
import requests
import yaml
import types
import pandas as pd
from pandas import DataFrame as d
import traceback

func_name = ""
tg_messaging = True
func_name_print = False


class Tg_messenger:
    token = json.load(
        open("../access/telegram.json", "r")
    )
    chat = str(226989883)
    url = "https://api.telegram.org/bot"

    def __call__(self, message, locals=False):
        if locals:
            message = {
                key: value
                for key, value in message.items()
                if not key.startswith("__")
                and isinstance(
                    value, (str, int, dict, list)
                )
            }
            for key, value in message.copy().items():
                if isinstance(value, Exception):
                    message["Exception"] = message.pop(
                        key
                    ).args

        message = "```\n" + yaml.dump(message) + "```"
        _ = [
            self.url,
            self.token,
            "/sendMessage?chat_id=",
            self.chat,
            "&parse_mode=Markdown&text=",
            message,
        ]
        _ = "".join(_)

        ans = requests.get(_)

        # return ans.json()


tg_message = Tg_messenger()


def tg_dec(function):
    @wraps(function)
    def wrapper_accepting_arguments(*args, **kwargs):
        # todo : implement https://stackoverflow.com/questions/8977359/decorating-a-method-thats-already-a-classmethod
        try:
            global func_name, func_name_print
            if func_name_print:
                if func_name != function.__name__:
                    print(function.__name__)
                    func_name = function.__name__
            return function(*args, **kwargs)
        except Exception as e:
            if tg_messaging:
                tb = traceback.format_exc().splitlines()[
                    3:
                ]
                tb = tb[::-1]
                tg_message(
                    dict(args=args, kwargs=kwargs),
                )
                tg_message(dict(tb=tb))
            raise e

    return wrapper_accepting_arguments


def for_all_methods(decorator):
    def decorate(cls):
        print(cls.__name__)
        for (
            attr
        ) in (
            cls.__dict__
        ):  # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                setattr(
                    cls,
                    attr,
                    decorator(getattr(cls, attr)),
                )
        return cls

    return decorate


def pretty(
    t,
    indent=0,
    tab_sign=" ",
    depth=0,
    to_file: str = None,
):
    if to_file:
        f = open(to_file, "w")
        tmp_stdout = sys.stdout
        sys.stdout = f
    if not depth or depth > indent:
        if isinstance(t, dict):
            t = tuple(t.items())
        elif isinstance(t, (set, list)):
            t = tuple(t)
        if isinstance(t, (int, str)):
            print(indent * tab_sign + str(t))
        elif isinstance(t, tuple):
            short_t = len(t) == 2 and isinstance(
                t[1], (int, str)
            )
            for i, ele in enumerate(t):
                pretty(
                    ele,
                    indent + 1 + (i * short_t),
                    tab_sign,
                    depth,
                )
        else:
            try:
                return pretty(
                    list(t),
                    indent,
                    tab_sign,
                    depth,
                    to_file,
                )
            except:
                raise TypeError("type not supported")
    if to_file:
        sys.stdout = tmp_stdout
        print("export complete")


def populate_kwargs(*args: tuple):
    """
    requires tuple['name','value',]
    allows tuple['name','value',('default')]
    """
    kwargs = {}
    for ele in args:
        name = ele[0]
        value = ele[1]
        default = ele[2] if len(ele) > 2 else None
        if value:
            kwargs[name] = value
        elif default:
            kwargs[name] = default
    return kwargs


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        retval = func(*args, **kwargs)
        finish = time.time()
        print(
            func.__name__,
            "took",
            round(finish - start),
            "seconds to run",
        )
        return retval

    return wrapper


@for_all_methods(tg_dec)
class Extract:
    """
    E of ETL.
    init kwargs are passed in every get call (request).
    """

    def __init__(
        self, url: str, data: dict = None, **kwargs,
    ):
        self.url = url
        self.data = data
        # params for every request
        self.params = {**kwargs}

    def extract(
        self,
        endpoint: str = "",
        method: str = "GET",
        data: dict = None,
        params: dict = None,
        all_kw_to_params: bool = True,
        **kwargs,
    ):
        """
        get kwargs are passed in get call
        """
        kwargs2 = dict(
            method=method,
            url=self.url + endpoint,
            data=data,
            params=params,
        )
        if all_kw_to_params:
            kwargs2["params"] = {
                **self.params,
                **kwargs,
            }
        else:
            kwargs2["data"] = {
                **self.params,
                **kwargs,
            }
        if data is not None:
            kwargs2["data"] = {
                **data,
                **kwargs2["data"],
            }
        if params is not None:
            kwargs2["params"] = {
                **params,
                **kwargs2["params"],
            }
        kwargs2 = {
            k: v
            for k, v in kwargs2.items()
            if v is not None
        }
        ans = requests.request(**kwargs2)
        try:
            ans = ans.json()
        except:
            print("wtf")
        return ans

    @staticmethod
    def write(filename: str, ans):
        pickle.dump(ans, open(filename, "wb"))

    @staticmethod
    def read(filename: str):
        # datatowalk: gc_exp = Loader.read('getcourse.json')
        if ".json" in filename:
            ans = open(filename).read()
            ans = json.loads(ans)
        else:
            ans = pickle.load(open(filename, "rb"))
        return ans


@for_all_methods(tg_dec)
class Transform:
    """
    cols_mapping         : list of dicts for enrich
    cols_naming          : list, dict for drop and order
    date_cols_mapping    : dict for date_mapping
    ---
    example
    ---
    cols_naming = [] or {:}
    date_cols_mapping = {
    "date_mapped": {
        **{
            timedelta(days=ele): 13
            if ele > 3
            else 6
            for ele in range(7)
        },
        "from": "date",
        }
    }
    date_cols_mapping = {
        "date_mapped": {
            True: 13,
            False: 6,
            "check_gt": timedelta(days=5, hours=11),
            "from": "Дата создания",
        }
    }

    """

    date_cols_mapping = None
    cols_naming = None
    cols_mapping = None
    date_out_format = "%Y-%m-%d"
    date_mapped_out_format = "%Y-%m-%d"
    date_in_format = None

    def change(self, data):
        return data

    def save(self, data, save_postfix):
        _ = type(self).__name__[:-2]
        if save_postfix is not None:
            _ += "_" + save_postfix
        Extract.write(
            f"../exports/{_}.p", data,
        )

    def date_mapping(
        self, a_date: datetime, date_map: dict
    ):
        weekday = a_date.weekday()
        week_start = a_date.normalize() - timedelta(
            days=weekday
        )

        weekday = timedelta(weekday)
        diff = date_map[weekday]
        diff = timedelta(days=diff)
        return week_start + diff

    def format_dates(
        self, data: pd.DataFrame, logic: str
    ):
        # logic=['in','out']
        cols = list(data.columns)
        date_cols = [
            e
            for e in cols
            if e.startswith(("date", "Дата", "day"))
        ]
        for ele in date_cols:
            try:
                if logic == "in":
                    data[ele] = pd.to_datetime(
                        data[ele],
                        format=self.date_in_format,
                    )
                if logic == "out":
                    if "mapped" in ele:
                        data[ele] = data[
                            ele
                        ].dt.strftime(
                            self.date_mapped_out_format
                        )
                    else:
                        data[ele] = data[
                            ele
                        ].dt.strftime(
                            self.date_out_format
                        )
            except:
                pass
        return data

    def drop_and_order(self, data, cols_mapping):
        if isinstance(cols_mapping, dict):
            for k, v in cols_mapping.items():
                if v == "":
                    cols_mapping[k] = k
        if isinstance(cols_mapping, list):
            cols_mapping = {e: e for e in cols_mapping}
        cols_naming = []
        cols_filter = [
            (k, cols_naming.append(v))[0]
            for k, v in cols_mapping.items()
            if k in list(data.columns)
        ]
        data = data[cols_filter]
        data.columns = cols_naming
        return data

    def enrich_by_dict(self, data):
        # this enriches by dict
        for k, v in self.cols_mapping.items():
            data[k] = np.nan
            for k1, v1 in v.items():
                for ele in v1:
                    filter_rows = df_filter(data, ele)
                    data.loc[filter_rows, k] = k1

        data = data.fillna("")
        return data

    def enrich(self, data):
        # this is sequential
        for ele in self.cols_mapping:
            if ele["col"] not in data.columns:
                data[ele["col"]] = np.nan
            filter_rows = df_filter(data, ele["check"])
            data.loc[filter_rows, ele["col"]] = ele[
                "result"
            ]
        data = data.fillna("")
        return data

    @staticmethod
    def finalize(data: pd.DataFrame):
        data = data.fillna("")
        cols = data.columns
        cols = (
            cols.to_series()
            .apply(
                lambda x: x
                if isinstance(x, str)
                else "_".join(x)
            )
            .tolist()
        )

        retval = [
            [str(e1) for e1 in e]
            for e in data.to_numpy()
        ]
        retval.insert(0, cols)
        return retval

    @staticmethod
    def undo_finalize(data: list):

        return pd.DataFrame(data[1:], columns=data[0])

    def transform(self, data, save_postfix=None):
        if isinstance(self.cols_naming, list):
            self.cols_naming = {
                e: e for e in self.cols_naming
            }
        elif isinstance(self.cols_naming, dict):
            for k, v in self.cols_naming.copy().items():
                if v == "":
                    self.cols_naming[k] = k

        data = self.change(data)
        data = data.fillna("")
        data = self.format_dates(data, "in")
        if self.cols_mapping is not None:
            self.cols_naming.update(
                {
                    e1: e1
                    for e1 in set(
                        e["col"]
                        for e in self.cols_mapping
                    )
                }
            )
            data = self.enrich(data)
        if self.date_cols_mapping is not None:
            for k, v in self.date_cols_mapping.items():
                data[k] = data[v["from"]].apply(
                    self.date_mapping, args=(v,)
                )

        if self.cols_naming is not None:
            data = self.drop_and_order(
                data, self.cols_naming
            )
        data = self.format_dates(data, "out")
        self.save(data, save_postfix)
        data = Transform.finalize(data)
        return data


@for_all_methods(tg_dec)
class Load:
    def __init__(
        self,
        worksheet_name: str,
        spreadsheet_name: str = "Нейрософия.Выгрузка",
    ):
        self.spreadsheet_name = spreadsheet_name
        self.spreadsheet = None
        self.worksheet_name = worksheet_name
        self.worksheet = None
        self.ga = g.service_account(
            "../access/google.json"
        )

    def load(
        self,
        data: list,
        raw: bool = False,
        drop: bool = True,
        freeze: list = [1, 1],
    ):
        self.spreadsheet = self.ga.open(
            self.spreadsheet_name
        )
        rows = len(data)
        cols = len(data[0])
        size = rows * cols
        speed = 0.00007
        print(f"loading {rows}x{cols}={size:_}")
        print(f"estimated {round(10+size*speed,0)} sec")
        if drop:
            try:
                self.worksheet = self.spreadsheet.worksheet(
                    self.worksheet_name
                )
                self.spreadsheet.del_worksheet(
                    self.worksheet
                )
            except:
                pass
            self.spreadsheet.add_worksheet(
                self.worksheet_name,
                rows=rows,
                cols=cols,
            )
        self.worksheet = self.spreadsheet.worksheet(
            self.worksheet_name
        )
        step = 4096 if rows > 4096 else rows
        for row in range(0, rows, step):
            range_name = (
                f""
                f"{rowcol_to_a1(row+1,0+1)}"
                f":{rowcol_to_a1(row +step+1,cols+1)}"
            )
            ans = self.worksheet.update(
                range_name,
                data[row : row + step],
                raw=raw,
            )

        self.worksheet.set_basic_filter()
        try:
            gsf.set_frozen(
                self.worksheet,
                rows=freeze[1],
                cols=freeze[0],
            )
        except:
            pass
        return ans

    def check_files_are_fresh(self, conditions: dict):
        # todo: maybe check on extract
        pass

    def onerror_mailer(self):
        # todo: fb ?error rates are high
        pass

    def report_update(self):
        pass


today = datetime.today().strftime("%Y-%m-%d")
year_start = today[:4] + "-01-01"


def unnesting_json(df, explode, axis=1):
    """
    example

    x = pd.DataFrame(ans[1:], columns=ans[0])
    a=unnesting(x[['subscriptions','vk_user_id']],['subscriptions'])

    but it feels like json_normalize
    """
    if axis == 1:
        idx = df.index.repeat(df[explode[0]].str.len())
        df1 = pd.concat(
            [
                pd.DataFrame(
                    {x: np.concatenate(df[x].values)}
                )
                for x in explode
            ],
            axis=1,
        )
        df1.index = idx

        return df1.join(df.drop(explode, 1), how="left")
    else:
        df1 = pd.concat(
            [
                pd.DataFrame(
                    df[x].tolist(), index=df.index
                ).add_prefix(x)
                for x in explode
            ],
            axis=1,
        )
        return df1.join(df.drop(explode, 1), how="left")


def df_filter(df, condlist):
    # example: df.loc[df_filter(df,condlist)]
    # !!! use only df.loc[ans]
    # condlist for conditions united with AND
    # cond for 1 condition
    def df_f_base(df, cond):
        if isinstance(cond, dict):
            cond = [
                cond["type"],
                cond["col"],
                cond["value"],
            ]
        if cond[0] == "contains":
            if isinstance(cond[2], (str, int)):
                cond[2] = [cond[2]]

            a = df[cond[1]].str.contains(
                "|".join(cond[2]),
                regex=True,
                case=False,
            )

        if cond[0] == "isin":
            a = df[cond[1]].isin(cond[2])
        if cond[0] == "isna":
            a = df[cond[1]].isna()
        if cond[0] == "neq":
            a = df[cond[1]] != cond[2]
        if cond[0] == "eq":
            a = df[cond[1]] == cond[2]

        return df[a].index

    if isinstance(condlist[0], str):
        condlist = [condlist]
    for ele in condlist:
        df = df.loc[df_f_base(df, ele)]

    return df.index


def debug_dec(function):
    def wrapper_accepting_arguments(*args, **kwargs):
        ans = function(*args, **kwargs)
        print("ans")
        pretty(ans.json())
        print("request")
        pretty(kwargs)
        pretty(args)
        return ans

    return wrapper_accepting_arguments


def pd_list_cols(df: pd.DataFrame):
    return list(df.columns)


def pd_count_values(df: pd.DataFrame, subset=False):
    if not subset:
        return dict(df.value_counts())
    return dict(df.value_counts(subset=subset))


def pd_mapped_cols(df: pd.DataFrame):
    return [
        e for e in pd_list_cols(df) if "mapped" in e
    ]


def pd_list_unique(df: pd.DataFrame):
    return list(df.unique())


def best_pprint(a):
    yaml.dump(a)

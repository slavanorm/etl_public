# todo add cfg reader into extract class
import yaml

yaml.warnings({"YAMLLoadWarning": False})


from main.fb import FB_e, FB_t

e = FB_e()
df1 = e.__dict__


df2 = {
    key: value
    for key, value in FB_t.__dict__.items()
    if not key.startswith("__") and not callable(value)
}

df1 = dict(extract=df1, transform=df2)

filename = "../config/FB.yaml"
with open(filename, "w") as f:
    yaml.dump(df1, f)

"""
with open(filename, "r") as f:
    ans = yaml.unsafe_load(f,Loader=yaml.fullLoader)
"""

v = 1

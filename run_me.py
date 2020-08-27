from main.base import tg_message
from main.fb import run_fb
from main.getcourse import run_gc
from main.vk import run_vk
from main.senler import run_se
from main.pivot import run_pivot
import schedule
from time import sleep


times_to_run = ["00:00", "12:00", "20:00"]

mute = False

if mute:
    import main.base

    main.base.tg_messaging = False


def full_run(from_pickle=False):

    tg_message("Neurosofia job started")
    run_fb(from_pickle=from_pickle)
    run_vk()  # is fast
    run_gc(from_pickle=from_pickle)
    run_se(from_pickle=from_pickle)
    run_pivot()  # is fast
    tg_message("Neurosofia job complete")


for time in times_to_run:
    schedule.every().day.at(time).do(full_run)

while True:
    schedule.run_pending()
    sleep(100)

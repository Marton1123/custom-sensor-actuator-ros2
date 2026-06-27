"""Arranque mínimo para M5Stack UnitV sin PMU, LCD ni modelo."""

import gc
import time

gc.collect()
# Da tiempo para interrumpir el arranque desde el REPL si main.py falla.
time.sleep_ms(5000)

with open("/flash/main.py", "r") as main_file:
    exec(main_file.read())

import lgpio
h = lgpio.gpiochip_open(4)
lgpio.gpio_claim_output(h, 18, 0)
lgpio.gpio_claim_output(h, 23, 0)
lgpio.gpiochip_close(h)
print("Motor detenido con exito (pines 18 y 23 a 0).")

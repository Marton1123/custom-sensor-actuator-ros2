import sys
import time
import lgpio

# Pines conectados al L298N
M1_IN1 = 18
M1_IN2 = 23
M2_IN3 = 25
M2_IN4 = 26

if len(sys.argv) < 2:
    print("Uso: python3 calibrar_dos_motores.py <tiempo_en_segundos> [direccion (1/2)]")
    print("Ejemplo para probar 3.7 segundos en dirección 1 (Botellas):")
    print("  python3 calibrar_dos_motores.py 3.7 1")
    print("Ejemplo para probar 3.7 segundos en dirección 2 (Latas):")
    print("  python3 calibrar_dos_motores.py 3.7 2")
    sys.exit(1)

try:
    tiempo = float(sys.argv[1])
except ValueError:
    print("Error: El tiempo debe ser un número decimal (ej. 3.7)")
    sys.exit(1)

direccion = sys.argv[2] if len(sys.argv) > 2 else "1"

try:
    h = lgpio.gpiochip_open(4)
    print("gpiochip4 abierto correctamente.")
except Exception as e:
    print(f"Error al abrir GPIO: {e}")
    sys.exit(1)

pines = [M1_IN1, M1_IN2, M2_IN3, M2_IN4]

# Configurar pines
for p in pines:
    lgpio.gpio_claim_output(h, p, 0)

# Asegurar que empiecen apagados
for p in pines:
    lgpio.gpio_write(h, p, 0)

print(f"Arrancando en 1 segundo... Girando ambos motores (Dirección {direccion}) por {tiempo}s")
time.sleep(1.0)

if direccion == "1":
    lgpio.gpio_write(h, M1_IN1, 1)
    lgpio.gpio_write(h, M1_IN2, 0)
    lgpio.gpio_write(h, M2_IN3, 1)
    lgpio.gpio_write(h, M2_IN4, 0)
else:
    lgpio.gpio_write(h, M1_IN1, 0)
    lgpio.gpio_write(h, M1_IN2, 1)
    lgpio.gpio_write(h, M2_IN3, 0)
    lgpio.gpio_write(h, M2_IN4, 1)

time.sleep(tiempo)

# Frenar motores
for p in pines:
    lgpio.gpio_write(h, p, 0)
    lgpio.gpio_free(h, p)

lgpio.gpiochip_close(h)
print("¡Giro completado y motores apagados!")

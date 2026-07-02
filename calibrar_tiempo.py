import sys
import time
import lgpio

# Pines conectados al L298N
IN1_PIN = 18
IN2_PIN = 23

if len(sys.argv) < 2:
    print("Uso: python3 calibrar_tiempo.py <tiempo_en_segundos> [direccion (der/izq)]")
    print("Ejemplo para probar 1.5 segundos a la derecha:")
    print("  python3 calibrar_tiempo.py 1.5 der")
    sys.exit(1)

try:
    tiempo = float(sys.argv[1])
except ValueError:
    print("Error: El tiempo debe ser un número decimal (ej. 1.25)")
    sys.exit(1)

direccion = sys.argv[2] if len(sys.argv) > 2 else "der"

try:
    h = lgpio.gpiochip_open(4)
except Exception as e:
    print(f"Error al abrir GPIO: {e}")
    sys.exit(1)

# Configurar pines
lgpio.gpio_claim_output(h, IN1_PIN)
lgpio.gpio_claim_output(h, IN2_PIN)

# Asegurar que empiece apagado
lgpio.gpio_write(h, IN1_PIN, 0)
lgpio.gpio_write(h, IN2_PIN, 0)

print(f"Arrancando en 1 segundo... Girando hacia {direccion.upper()} por {tiempo}s")
time.sleep(1.0)

if direccion.lower() == "der":
    lgpio.gpio_write(h, IN1_PIN, 1)
    lgpio.gpio_write(h, IN2_PIN, 0)
else:
    lgpio.gpio_write(h, IN1_PIN, 0)
    lgpio.gpio_write(h, IN2_PIN, 1)

time.sleep(tiempo)

# Frenar motor
lgpio.gpio_write(h, IN1_PIN, 0)
lgpio.gpio_write(h, IN2_PIN, 0)

# Liberar recursos
lgpio.gpio_free(h, IN1_PIN)
lgpio.gpio_free(h, IN2_PIN)
lgpio.gpiochip_close(h)

print("¡Giro completado y motor apagado!")

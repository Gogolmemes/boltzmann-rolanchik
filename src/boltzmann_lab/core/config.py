POTENTIAL_POINTS = 1024
HIST_BINS = 40

FIELD_WIDTH = 900
FIELD_HEIGHT = 280

# Физическое поле имеет те же пропорции, что и картинка.
WORLD_H = 1.0
WORLD_W = FIELD_WIDTH / FIELD_HEIGHT

FPS = 60
SUBSTEPS = 4
DT = 1.0 / (FPS * SUBSTEPS)

BARRIER_MIN = 0.10          # допустимый диапазон положения перегородки
BARRIER_MAX = 0.90
BARRIER_MAX_SPEED = 4.0     # макс. скорость стенки, в единицах поля в секунду
BARRIER_RESPONSE = 40.0     # 1/с: насколько быстро стенка догоняет курсор

DEMON_COLORS = [
    (239, 68, 68), 
    (34, 197, 94), 
    (59, 130, 246)
]
DEMON_HEX = [
    "#ef4444",
    "#22c55e", 
    "#3b82f6"
]


MAX_TEMPERATURE = 50.0 # Максимальная температура, превышение которой приводит к аварийному завершению симуляции.
TEMPERATURE_CHECK_INTERVAL = 1 # Интервал проверки температуры (в секундах)
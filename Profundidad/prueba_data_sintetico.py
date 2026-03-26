import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Los datos proporcionados
raw_data = """2026-03-25 11:41:14 | Clase: GRANDE | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 168.0mm | Borde: 152.4mm | Centro: 159.0mm | Delta: 6.6mm | Ratio: 0.907 | Score: 0.378
2026-03-25 11:41:21 | Clase: MEDIANA | Estado: DUDA=REAL ABOLLADA | H_Max: 113.0mm | Borde: 105.0mm | Centro: 104.0mm | Delta: 1.0mm | Ratio: 0.930 | Score: 0.088
2026-03-25 11:41:27 | Clase: CHICA | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 26.0mm | Borde: 20.3mm | Centro: 20.4mm | Delta: 0.1mm | Ratio: 0.780 | Score: 0.117
2026-03-25 11:41:34 | Clase: GRANDE | Estado: DUDA=REAL ABOLLADA | H_Max: 159.0mm | Borde: 152.3mm | Centro: 156.4mm | Delta: 4.1mm | Ratio: 0.958 | Score: 0.226
2026-03-25 11:41:41 | Clase: MEDIANA | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 114.0mm | Borde: 104.5mm | Centro: 108.5mm | Delta: 4.0mm | Ratio: 0.917 | Score: 0.242
2026-03-25 11:41:48 | Clase: CHICA | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 28.0mm | Borde: 22.2mm | Centro: 21.6mm | Delta: 0.6mm | Ratio: 0.792 | Score: 0.135
2026-03-25 11:41:54 | Clase: GRANDE | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 165.0mm | Borde: 152.6mm | Centro: 159.9mm | Delta: 7.3mm | Ratio: 0.925 | Score: 0.403
2026-03-25 11:42:08 | Clase: CHICA | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 28.0mm | Borde: 20.0mm | Centro: 23.0mm | Delta: 3.0mm | Ratio: 0.714 | Score: 0.293
2026-03-25 11:42:19 | Clase: GRANDE | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 168.0mm | Borde: 152.4mm | Centro: 159.0mm | Delta: 6.6mm | Ratio: 0.907 | Score: 0.378
2026-03-25 11:42:25 | Clase: MEDIANA | Estado: DUDA=REAL ABOLLADA | H_Max: 113.0mm | Borde: 105.0mm | Centro: 104.0mm | Delta: 1.0mm | Ratio: 0.930 | Score: 0.088
2026-03-25 11:42:32 | Clase: CHICA | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 26.0mm | Borde: 20.3mm | Centro: 20.4mm | Delta: 0.1mm | Ratio: 0.780 | Score: 0.117
2026-03-25 11:42:39 | Clase: GRANDE | Estado: DUDA=REAL ABOLLADA | H_Max: 159.0mm | Borde: 152.3mm | Centro: 156.4mm | Delta: 4.1mm | Ratio: 0.958 | Score: 0.226
2026-03-25 11:42:46 | Clase: MEDIANA | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 114.0mm | Borde: 104.5mm | Centro: 108.5mm | Delta: 4.0mm | Ratio: 0.917 | Score: 0.242
2026-03-25 11:42:53 | Clase: CHICA | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 28.0mm | Borde: 22.2mm | Centro: 21.6mm | Delta: 0.6mm | Ratio: 0.792 | Score: 0.135
2026-03-25 11:42:59 | Clase: GRANDE | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 165.0mm | Borde: 152.6mm | Centro: 159.9mm | Delta: 7.3mm | Ratio: 0.925 | Score: 0.403
2026-03-25 11:43:13 | Clase: CHICA | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 28.0mm | Borde: 20.0mm | Centro: 23.0mm | Delta: 3.0mm | Ratio: 0.714 | Score: 0.293
2026-03-25 11:43:23 | Clase: GRANDE | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 168.0mm | Borde: 152.4mm | Centro: 159.0mm | Delta: 6.6mm | Ratio: 0.907 | Score: 0.378
2026-03-25 11:43:30 | Clase: MEDIANA | Estado: DUDA=REAL ABOLLADA | H_Max: 113.0mm | Borde: 105.0mm | Centro: 104.0mm | Delta: 1.0mm | Ratio: 0.930 | Score: 0.088
2026-03-25 11:43:37 | Clase: CHICA | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 26.0mm | Borde: 20.3mm | Centro: 20.4mm | Delta: 0.1mm | Ratio: 0.780 | Score: 0.117
2026-03-25 11:43:44 | Clase: GRANDE | Estado: DUDA=REAL ABOLLADA | H_Max: 159.0mm | Borde: 152.3mm | Centro: 156.4mm | Delta: 4.1mm | Ratio: 0.958 | Score: 0.226
2026-03-25 11:43:51 | Clase: MEDIANA | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 114.0mm | Borde: 104.5mm | Centro: 108.5mm | Delta: 4.0mm | Ratio: 0.917 | Score: 0.242
2026-03-25 11:43:58 | Clase: CHICA | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 28.0mm | Borde: 22.2mm | Centro: 21.6mm | Delta: 0.6mm | Ratio: 0.792 | Score: 0.135
2026-03-25 11:44:04 | Clase: GRANDE | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 165.0mm | Borde: 152.6mm | Centro: 159.9mm | Delta: 7.3mm | Ratio: 0.925 | Score: 0.403
2026-03-25 11:44:18 | Clase: CHICA | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 28.0mm | Borde: 20.0mm | Centro: 23.0mm | Delta: 3.0mm | Ratio: 0.714 | Score: 0.293
2026-03-25 11:44:28 | Clase: GRANDE | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 168.0mm | Borde: 152.4mm | Centro: 159.0mm | Delta: 6.6mm | Ratio: 0.907 | Score: 0.378
2026-03-25 11:47:53 | Clase: GRANDE | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 165.0mm | Borde: 151.1mm | Centro: 156.0mm | Delta: 4.9mm | Ratio: 0.916 | Score: 0.286
2026-03-25 11:47:57 | Clase: MEDIANA | Estado: DUDA=REAL ABOLLADA | H_Max: 114.0mm | Borde: 103.8mm | Centro: 105.0mm | Delta: 1.2mm | Ratio: 0.910 | Score: 0.107
2026-03-25 11:48:01 | Clase: CHICA | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 26.0mm | Borde: 19.6mm | Centro: 22.4mm | Delta: 2.7mm | Ratio: 0.755 | Score: 0.259
2026-03-25 11:48:05 | Clase: GRANDE | Estado: DUDA=REAL ABOLLADA | H_Max: 157.0mm | Borde: 151.5mm | Centro: 149.0mm | Delta: 2.5mm | Ratio: 0.965 | Score: 0.144
2026-03-25 11:48:13 | Clase: CHICA | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 31.0mm | Borde: 23.4mm | Centro: 21.0mm | Delta: 2.4mm | Ratio: 0.755 | Score: 0.243
2026-03-25 11:48:17 | Clase: GRANDE | Estado: DUDA=REAL ABOLLADA | H_Max: 161.0mm | Borde: 152.4mm | Centro: 151.6mm | Delta: 0.8mm | Ratio: 0.947 | Score: 0.065
2026-03-25 11:48:22 | Clase: CHICA | Estado: ABOLLADA=REAL ABOLLADA | H_Max: 28.0mm | Borde: 21.2mm | Centro: 22.0mm | Delta: 0.8mm | Ratio: 0.759 | Score: 0.158
2026-03-25 11:44:35 | Clase: GRANDE | Estado: DUDA | H_Max: 154.0mm | Borde: 150.3mm | Centro: 153.0mm | Delta: 2.7mm | Ratio: 0.976 | Score: 0.148
2026-03-25 11:44:41 | Clase: MEDIANA | Estado: DUDA | H_Max: 107.0mm | Borde: 103.3mm | Centro: 107.0mm | Delta: 3.7mm | Ratio: 0.965 | Score: 0.203
2026-03-25 11:44:48 | Clase: CHICA | Estado: ABOLLADA | H_Max: 28.0mm | Borde: 20.4mm | Centro: 22.4mm | Delta: 2.0mm | Ratio: 0.729 | Score: 0.233
2026-03-25 11:44:55 | Clase: GRANDE | Estado: DUDA | H_Max: 154.0mm | Borde: 150.2mm | Centro: 154.0mm | Delta: 3.8mm | Ratio: 0.976 | Score: 0.200
2026-03-25 11:45:02 | Clase: MEDIANA | Estado: DUDA | H_Max: 106.0mm | Borde: 102.4mm | Centro: 106.0mm | Delta: 3.6mm | Ratio: 0.966 | Score: 0.198
2026-03-25 11:45:09 | Clase: CHICA | Estado: ABOLLADA | H_Max: 29.0mm | Borde: 21.4mm | Centro: 22.4mm | Delta: 1.0mm | Ratio: 0.739 | Score: 0.181
2026-03-25 11:45:21 | Clase: GRANDE | Estado: DUDA | H_Max: 154.0mm | Borde: 150.3mm | Centro: 153.0mm | Delta: 2.7mm | Ratio: 0.976 | Score: 0.148
2026-03-25 11:45:27 | Clase: MEDIANA | Estado: DUDA | H_Max: 107.0mm | Borde: 103.3mm | Centro: 107.0mm | Delta: 3.7mm | Ratio: 0.965 | Score: 0.203
2026-03-25 11:45:34 | Clase: CHICA | Estado: ABOLLADA | H_Max: 28.0mm | Borde: 20.4mm | Centro: 22.4mm | Delta: 2.0mm | Ratio: 0.729 | Score: 0.233
2026-03-25 11:45:41 | Clase: GRANDE | Estado: DUDA | H_Max: 154.0mm | Borde: 150.2mm | Centro: 154.0mm | Delta: 3.8mm | Ratio: 0.976 | Score: 0.200
2026-03-25 11:45:48 | Clase: MEDIANA | Estado: DUDA | H_Max: 106.0mm | Borde: 102.4mm | Centro: 106.0mm | Delta: 3.6mm | Ratio: 0.966 | Score: 0.198
2026-03-25 11:45:57 | Clase: GRANDE | Estado: DUDA | H_Max: 155.0mm | Borde: 151.3mm | Centro: 155.4mm | Delta: 4.1mm | Ratio: 0.976 | Score: 0.217
2026-03-25 11:46:04 | Clase: GRANDE | Estado: DUDA | H_Max: 156.0mm | Borde: 152.7mm | Centro: 151.0mm | Delta: 1.7mm | Ratio: 0.979 | Score: 0.096
2026-03-25 11:46:11 | Clase: CHICA | Estado: DUDA | H_Max: 25.0mm | Borde: 20.4mm | Centro: 23.0mm | Delta: 2.6mm | Ratio: 0.817 | Score: 0.220
2026-03-25 11:46:17 | Clase: MEDIANA | Estado: NORMAL | H_Max: 106.0mm | Borde: 102.1mm | Centro: 103.0mm | Delta: 0.9mm | Ratio: 0.963 | Score: 0.065
2026-03-25 11:46:23 | Clase: MEDIANA | Estado: ABOLLADA | H_Max: 110.0mm | Borde: 103.6mm | Centro: 110.0mm | Delta: 6.4mm | Ratio: 0.942 | Score: 0.347
2026-03-25 11:46:30 | Clase: GRANDE | Estado: ABOLLADA | H_Max: 154.0mm | Borde: 151.6mm | Centro: 93.2mm | Delta: 58.3mm | Ratio: 0.984 | Score: 2.923
2026-03-25 11:46:37 | Clase: CHICA | Estado: ABOLLADA | H_Max: 26.0mm | Borde: 19.8mm | Centro: 23.0mm | Delta: 3.2mm | Ratio: 0.762 | Score: 0.279
2026-03-25 11:46:44 | Clase: CHICA | Estado: ABOLLADA | H_Max: 27.0mm | Borde: 20.4mm | Centro: 23.0mm | Delta: 2.6mm | Ratio: 0.754 | Score: 0.255
2026-03-25 11:46:54 | Clase: GRANDE | Estado: DUDA | H_Max: 155.0mm | Borde: 151.3mm | Centro: 155.4mm | Delta: 4.1mm | Ratio: 0.976 | Score: 0.217
2026-03-25 11:47:01 | Clase: GRANDE | Estado: DUDA | H_Max: 156.0mm | Borde: 152.7mm | Centro: 151.0mm | Delta: 1.7mm | Ratio: 0.979 | Score: 0.096
2026-03-25 11:47:07 | Clase: CHICA | Estado: DUDA | H_Max: 25.0mm | Borde: 20.4mm | Centro: 23.0mm | Delta: 2.6mm | Ratio: 0.817 | Score: 0.220
2026-03-25 11:47:14 | Clase: MEDIANA | Estado: NORMAL | H_Max: 106.0mm | Borde: 102.1mm | Centro: 103.0mm | Delta: 0.9mm | Ratio: 0.963 | Score: 0.065
2026-03-25 11:47:20 | Clase: MEDIANA | Estado: ABOLLADA | H_Max: 110.0mm | Borde: 103.6mm | Centro: 110.0mm | Delta: 6.4mm | Ratio: 0.942 | Score: 0.347
2026-03-25 11:47:27 | Clase: GRANDE | Estado: ABOLLADA | H_Max: 154.0mm | Borde: 151.6mm | Centro: 93.2mm | Delta: 58.3mm | Ratio: 0.984 | Score: 2.923
2026-03-25 11:47:34 | Clase: CHICA | Estado: ABOLLADA | H_Max: 26.0mm | Borde: 19.8mm | Centro: 23.0mm | Delta: 3.2mm | Ratio: 0.762 | Score: 0.279
2026-03-25 11:47:41 | Clase: CHICA | Estado: ABOLLADA | H_Max: 27.0mm | Borde: 20.4mm | Centro: 23.0mm | Delta: 2.6mm | Ratio: 0.754 | Score: 0.255
2026-03-25 11:48:33 | Clase: GRANDE | Estado: DUDA | H_Max: 153.0mm | Borde: 152.3mm | Centro: 150.0mm | Delta: 2.3mm | Ratio: 0.995 | Score: 0.117
2026-03-25 11:48:40 | Clase: CHICA | Estado: ABOLLADA | H_Max: 28.0mm | Borde: 20.0mm | Centro: 21.0mm | Delta: 1.0mm | Ratio: 0.714 | Score: 0.193
2026-03-25 11:48:44 | Clase: GRANDE | Estado: DUDA | H_Max: 153.0mm | Borde: 151.7mm | Centro: 150.0mm | Delta: 1.7mm | Ratio: 0.991 | Score: 0.088
2026-03-25 11:48:47 | Clase: MEDIANA | Estado: DUDA | H_Max: 106.0mm | Borde: 101.4mm | Centro: 104.5mm | Delta: 3.1mm | Ratio: 0.957 | Score: 0.175
2026-03-25 11:48:51 | Clase: CHICA | Estado: ABOLLADA | H_Max: 25.0mm | Borde: 18.0mm | Centro: 24.4mm | Delta: 6.4mm | Ratio: 0.720 | Score: 0.459
2026-03-25 11:48:55 | Clase: GRANDE | Estado: NORMAL | H_Max: 153.0mm | Borde: 151.6mm | Centro: 152.0mm | Delta: 0.4mm | Ratio: 0.991 | Score: 0.023
2026-03-25 11:48:58 | Clase: MEDIANA | Estado: NORMAL | H_Max: 107.0mm | Borde: 102.5mm | Centro: 103.0mm | Delta: 0.5mm | Ratio: 0.958 | Score: 0.047
2026-03-25 11:49:02 | Clase: CHICA | Estado: DUDA | H_Max: 28.0mm | Borde: 22.7mm | Centro: 24.0mm | Delta: 1.3mm | Ratio: 0.811 | Score: 0.160
2026-03-25 11:49:16 | Clase: GRANDE | Estado: DUDA | H_Max: 154.0mm | Borde: 150.8mm | Centro: 154.0mm | Delta: 3.2mm | Ratio: 0.979 | Score: 0.170
2026-03-25 11:49:19 | Clase: GRANDE | Estado: DUDA | H_Max: 154.0mm | Borde: 152.3mm | Centro: 154.0mm | Delta: 1.7mm | Ratio: 0.989 | Score: 0.091
2026-03-25 11:49:23 | Clase: MEDIANA | Estado: DUDA | H_Max: 107.0mm | Borde: 102.6mm | Centro: 106.0mm | Delta: 3.4mm | Ratio: 0.959 | Score: 0.192
2026-03-25 11:49:27 | Clase: CHICA | Estado: ABOLLADA | H_Max: 30.0mm | Borde: 23.4mm | Centro: 22.5mm | Delta: 0.9mm | Ratio: 0.780 | Score: 0.155
2026-03-25 11:49:31 | Clase: CHICA | Estado: ABOLLADA | H_Max: 28.0mm | Borde: 20.4mm | Centro: 23.4mm | Delta: 2.9mm | Ratio: 0.730 | Score: 0.283
2026-03-25 11:49:35 | Clase: GRANDE | Estado: ABOLLADA | H_Max: 154.0mm | Borde: 152.1mm | Centro: 0.0mm | Delta: 152.1mm | Ratio: 0.988 | Score: 7.612
2026-03-25 11:49:39 | Clase: MEDIANA | Estado: NORMAL | H_Max: 107.0mm | Borde: 101.4mm | Centro: 102.0mm | Delta: 0.6mm | Ratio: 0.948 | Score: 0.056
2026-03-25 11:49:43 | Clase: MEDIANA | Estado: NORMAL | H_Max: 108.0mm | Borde: 103.6mm | Centro: 105.0mm | Delta: 1.4mm | Ratio: 0.960 | Score: 0.088
2026-03-25 11:49:47 | Clase: CHICA | Estado: ABOLLADA | H_Max: 31.0mm | Borde: 23.8mm | Centro: 20.3mm | Delta: 3.5mm | Ratio: 0.768 | Score: 0.293

"""

# Nota: Para el script, pegamos el bloque de texto completo o leemos el archivo
def parse_latas(data_string):
    rows = []
    for line in data_string.strip().split('\n'):
        parts = line.split(' | ')
        # Extraer pares clave-valor
        d = {"Fecha": parts[0]}
        for p in parts[1:]:
            key, val = p.split(': ')
            # Limpiar notación científica y convertir a flotante si es posible
            try:
                d[key] = float(val)
            except ValueError:
                d[key] = val
        rows.append(d)
    return pd.DataFrame(rows)

# Cargar datos
df = parse_latas(raw_data)

# Configuración del gráfico
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Crear el scatter plot
# Usamos Delta en X y Ratio en Y para ver la agrupación
scatter = sns.scatterplot(
    data=df, 
    x='Delta', 
    y='Ratio', 
    hue='Estado', 
    style='Clase', 
    s=100, 
    palette={'NORMAL1=REAL ABOLLADA': '#2ecc71', 'ABOLLADA=REAL ABOLLADA': '#e74c3c', 'DUDA=REAL ABOLLADA':"#d0ff00",'NORMAL': "#001aff", 'ABOLLADA': "#f5c116", 'DUDA':"#586132"}
)

# Personalización
plt.title('Separación de Clases: Delta vs Ratio', fontsize=15)
plt.xlabel('Delta (Deformación mm)', fontsize=12)
plt.ylabel('Ratio (Integridad)', fontsize=12)
plt.legend(title='Estado / Clase', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
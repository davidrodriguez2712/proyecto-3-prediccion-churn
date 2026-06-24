## Reporte EDA - Churn Prediction
=== Reporte de Data Overview ===

Forma del dataset                          : 480000 filas | 15 columnas
Target                                     : "churn" -> Binaria (0, 1)
Cantidad de Features Numéricas             : 9
Cantidad de Features Categóricas           : 3
Feature del tiempo                         : "period"
Features Categóricas con alta cardinalidad : Ninguna
Features Categóricas con baja cardinalidad : "region", "payment_method", "contract_type"

=== Reporte de Missings ===

El dataset NO presenta ningún missing.
Se evaluó lo siguiente:
- Distribución de valores perdidos        : Sin resultados
- Correlación de nulidad entre variables  : Sin resultados 
- Cantidad de missing por feature         : Sin resultados

=== Reporte de Features Numéricas ===

Se puede observar lo siguiente:
- Existe relación monotónica creciente en las siguientes features: 
      - "support_calls"
      - "complaints_last_3m"
      - "late_payments"
- Las siguientes features tienen una cola hacia la derecha:
      - "marketing_emails_opened"
      - "support_calls"
      - "complaints_last_3m"
      - "late_payments"
- Pero al ser estas últimas variables discretas (conteos) y de baja cardinalidad,
no será necesaria ninguna transformación.

=== Reporte de Features Categóricas ===

Se identificó que aunque existe ligeros aumentos del churn rate promedio 
entre las categorías del mismo feature, no es un aumento considerable y tampoco
existe una relación ordinal clara para separarlo ordinalmente.

Además, la distribución de cada feature categórica muestra que no existen etiquetas 
o niveles raros a considerar. 

Posible features con OHE:
      - region
      - payment_method
      - contract_type

=== Reporte de Registros por Período y Churn por Período ===

Se observa que la cantidad de registros por período es mayor en los primeros meses 
y tiene una tendencia decreciente hasta el final del mismo.
    - Registros en Train   : 50.00%
    - Registros en Test    : 25.00%
    - Registros en OOT     : 25.00%

Respecto al Churn Rate por período, se observa que tanto el período de Train y Test
son muy similares en comportamiento con igual promedio global de ambos.
Por otro lado, el período propuesto como OOT (2026) sí varía ligeramente el promedio
global y tiene meses con un pico alto de Churn Rate y otros dos bastante bajos.
Se considerará esto al momento de analizar el monitoring y drift.

Por último, respecto a la distribución del Churn por período, se observa un cambio
en la prevalencia del OOT (-0.6pp vs Test | -0.53pp vs Train), al ser un cambio ligero,
se monitoreará principalmente en Calibration Drift.

=== Reporte de Análisis de Relaciones ===

Se evidencia con el análisis de Pearson que no encontramos una relación lineal 
entre las variables por pares ni contra el target.

Además, con Spearman vemos un comportamiento similar, puesto que no se encuentran indicios
de que exista una relación monotónica fuerte entre las features y el target.

Siguiendo con el análisis entre variables por pares, nos apoyamos utilizando Mutual Information
el cual también nos da indicios que no existen relaciones lineales o no lineales fuertes
entre algún feature y el target. Aunque si podemos mencionar que 'support_calls' es la variable
con mayor aporte de información vs las demás.

Ahora, mirando las relaciones multivariables lineales utilizamos el VIF, con el cual podemos 
interpretar de que existen 4 features con un VIF > 5, por lo que inferimos multicolinealidad
moderada en ellas. Se tomará en consideración para el feature engineering.

Por último, en la gráfica de Information Value podemos ver que ninguna feature por si sola logra
tener un poder predictivo débil, ya que están más cercanas a 0. Pero si es relevante mencionar
que la feature con mayor intuición de poder predictivo es 'support_calls'.

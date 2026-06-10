# Descripción de Datos

## Fuente de Datos

Dataset de comportamiento de clientes con:

- uso del servicio
- actividad
- quejas
- pagos
- features temporales

## Estructura Temporal

Datos indexados por periodo.

Se utiliza modelado temporal.

## Split Temporal

Train: 2023–2024  
Test: 2025  
OOT: 2026  

## Supuestos

- comportamiento predice churn
- actividad reciente es relevante
- puede existir drift temporal

## Prevención de Leakage

Las features usan solo información pasada.
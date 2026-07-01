# PB-CRL: Policy-Based Constrained Reinforcement Learning para la Cuenca Alta del Río Bogotá

Framework de RL con restricciones para la operación coordinada de los embalses Neusa, Sisga y Tominé,
con el objetivo de garantizar el caudal ecológico aguas abajo de la confluencia.

## Estructura

```
code_PBCRL/
├── src/pbcrl/
│   ├── data_contracts/   # Esquemas, contratos y validación de datos
│   ├── synthetic/        # Generador de datos sintéticos (reemplazable por datos reales)
│   ├── hydrology/        # Balance hídrico inverso
│   ├── stochastic/       # Modelo estocástico de afluencias (próximo paso)
│   └── environment/      # Entorno de RL (próximo paso)
└── tests/                # Pruebas unitarias
```

## Principio rector

Todo módulo recibe sus entradas como un `DataFrame` con esquema fijo definido en `data_contracts`.
Los datos sintéticos cumplen ese esquema; cuando lleguen los datos reales, basta limpiarlos
para que calcen con la interfaz, sin modificar el código.

## Instalación

```bash
pip install -r requirements.txt
```

## Pruebas

```bash
pytest tests/ -v
```

## Embalses

| Embalse | Capacidad útil (Mm³) | Área espejo (km²) |
|---------|---------------------|-------------------|
| Neusa   | ~88                 | ~10               |
| Sisga   | ~90                 | ~7                |
| Tominé  | ~690                | ~39               |

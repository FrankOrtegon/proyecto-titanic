from src.data_loader import DataLoader
from src.train_model import TitanicModel

def main():
    print("Entrenando modelo de Titanic...\n")

    loader = DataLoader("data/raw/Titanic-Dataset.csv")
    df = loader.cargar_datos()

    modelo = TitanicModel()
    metricas_modelo = modelo.entrenar(df)

    print("\n✅ Entrenamiento completado con éxito.")
    print(f"Métricas del modelo: ", metricas_modelo)
    print("Modelo guardado en la carpeta 'models/'")

if __name__ == "__main__":
    main()

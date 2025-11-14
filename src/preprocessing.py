import pandas as pd

class Preprocesador:
    """Simplifica los pasos de limpieza y transformación."""

    def preparar_datos(self, datos):
        
        print("Realiza procesamiento de los datos")

        # Copia de seguridad
        df = datos.copy()

        # Se computa datos nulos en la columna Age
        df['Age'] = df[['Age', 'Pclass']].apply(self.edad_media, axis=1)

        ## Eliminar la columna Cabin
        df.drop('Cabin', axis=1, inplace=True)

        ## Eliminar registros nulos en Embarked
        df = df.dropna(subset=['Embarked'])

        ## Eliminar columnas que son ID o registros unicos
        df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

        ## Convertir la variable Sex en dummy
        df_sex = pd.get_dummies(df['Sex'])

        df_sex['male'] = df_sex['male'].astype(int)
        df_sex['female'] = df_sex['female'].astype(int)

        ## Unir el df y la columna sex
        df = pd.concat([df, df_sex], axis=1)

        ## Convertir la variable Embarked en dummy
        df_embarked = pd.get_dummies(df['Embarked'])
        df_embarked['C'] = df_embarked['C'].astype(int)
        df_embarked['Q'] = df_embarked['Q'].astype(int)
        df_embarked['S'] = df_embarked['S'].astype(int)

        ## Unir el df y la columna Embarked
        df = pd.concat([df, df_embarked], axis = 1)

        ## Eliminar columnas Sex y Embarked
        df = df.drop(['Sex', 'Embarked'], axis=1)

        y = df['Survived']
        X = df.drop('Survived', axis=1)

        return X,y


    
    
    def edad_media(self, columnas):
        edad = columnas[0]
        pclass = columnas[1]

        if pd.isnull(edad):
          if pclass == 1:
            return 37
          elif pclass == 2:
            return 29
          else:
            return 24
        else:
          return edad
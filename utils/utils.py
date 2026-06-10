import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
import shap
import numpy as np

class PreprocessingData:
    def __init__(self):
        pass

    def split_type_features(self, dataset: pd.DataFrame):
        """Permite separar variables categóricas, numéricas y de tiempo"""

        numerical = ["int64", "float64"]
        categorical = ["object"]
        datetime = ["datetime64[ns]"]
        self.numerical_features = []
        self.categorial_features = []
        self.datetime_features = []
        self.wo_feature = []
        self.dataset_origin = dataset

        features_names = dataset.columns.to_list()
        #print(features_names)
        features_type = dataset.dtypes.astype("str").values.tolist()
        #print(features_type)

        for (feature_name, feature_type) in list(zip(features_names, features_type)):
            if feature_type in datetime:
                self.datetime_features.append(feature_name)
            if feature_type in numerical:
                self.numerical_features.append(feature_name)
            if feature_type in categorical:
                self.categorial_features.append(feature_name)

        return self.numerical_features, self.categorial_features, self.datetime_features
    
    def unique_values(self, dataset: pd.DataFrame):
        """Devuelve la cantidad de valores únicos para cada feature proporcionada"""
        features = dataset.columns.to_list()
        cant_unicos = []

        for i in features:
            unicos = len(dataset[i].unique().tolist())
            cant_unicos.append(unicos)

        reporte_unicos = {
            "feature": features,
            "cantidad de únicos": cant_unicos
        }

        reporte_unicos = pd.DataFrame(reporte_unicos).sort_values(by= "cantidad de únicos", ascending= False)

        return reporte_unicos

            
    def report_missings(self, dataset: pd.DataFrame):
        """Analiza la cantidad y porcentaje de missing en cada variable"""
        ## Columnas tentativas: Variable, Cant. No Nulos, Cant. Nulos, % Nulos
        
        features_col = dataset.columns.to_list()
        cant_nulos = [dataset[d].isna().sum() for d in features_col]
        mean_nulos = [dataset[d].isna().mean() for d in features_col]

        reporte = {
            "Columnas Tentativas": features_col,
            "Cant. Nulos": cant_nulos,
            "% Nulos": mean_nulos
        }

        reporte_df = pd.DataFrame(
            reporte,
        ).sort_values(by= "Cant. Nulos", ascending= False)

        fig, ax = plt.subplots()
        sns.heatmap(
            dataset.isna().transpose(),
            cmap= "YlGnBu",
            cbar_kws= {"label": "Valores perdidos"}
        )
        ax.set_title("Distribución de valores perdidos")
        plt.tight_layout()
        return reporte_df, fig, ax

    def limpieza_en_blancos(self):
        """Remueve espacios en blanco y saltos de línea al inicio y final de cada elemento del dataset"""
        dataset = self.dataset_origin

        for i in self.categorial_features:
            dataset[i] = dataset[i].astype("str").str.strip()
    
        return dataset
    
    def outliers(self, dataset: pd.DataFrame):
        """Visualizador de boxplot para cada feature numérica"""
        fig, axes = plt.subplots(len(self.numerical_features), 1, figsize = (10, 20))
        axes = axes.flatten()

        for i, ax in zip(self.numerical_features, axes):
            sns.boxplot(
                x = i,
                data= dataset,
                whis= 1.5,
                ax= ax
            )
            ax.set_title(f"Distribución en {i}")
        fig.tight_layout()
        fig.subplots_adjust(hspace= 0.6)

        return fig, axes
    
    def histogram(self, dataset: pd.DataFrame):
        """Muestra la distribución de cada feature numérico"""
        fig, axes = plt.subplots(len(self.numerical_features), 1, figsize= (6, 30))
        axes = axes.flatten()

        for i, ax in zip(self.numerical_features, axes):
            ax.hist(
                dataset[i],
                bins = "auto"
            )
            ax.set_title(f"{i}")

        return fig, axes

    def iqr_tecnica(self, dataset: pd.DataFrame):
        """Calcular numéricamente la cantidad de outliers con la técnica IQR"""
        
        numerical_features = []
        upper_list = []
        lower_list = []
        cantidad_outlier = []
        porcentaje_outlier = []
        remove_vals = ["late_payments", "marketing_emails_opened", "complaints_last_3m" ,"support_calls"]

        numerical_features_wo_payment = [i for i in self.numerical_features if i not in remove_vals]
        dataset_wins = dataset.copy()

        for i in numerical_features_wo_payment:
            
            Q1 = dataset[i].quantile(0.25)
            Q3 = dataset[i].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - (IQR * 1.5)
            upper = Q3 + (IQR * 1.5)
            mask_outliers = (dataset[i] < lower) | (dataset[i] > upper)
            cantidad_outliers = mask_outliers.sum()
            porcentaje_outliers = cantidad_outliers / (dataset[i].count())

            numerical_features.append(i)
            upper_list.append(upper)
            lower_list.append(lower)
            cantidad_outlier.append(cantidad_outliers)
            porcentaje_outlier.append(porcentaje_outliers)
            dataset_wins[i] = dataset_wins[i].clip(lower, upper)

        reporte_iqr = {
            "Feature Numérica": numerical_features,
            "Límite Inferior": lower_list,
            "Límite Superior": upper_list,
            "Cantidad de Outliers": cantidad_outlier,
            "Porcentaje Outlier": porcentaje_outlier
        }

        iqr_tabla = pd.DataFrame(reporte_iqr).sort_values(by= "Cantidad de Outliers", ascending= False)
        
        

        return iqr_tabla, dataset_wins
    
    def create_var_features(self, dataset: pd.DataFrame, features: list, mode = "numerico"):
        """Creación de las nuevas features/columnas que comparan moth over month (mom)"""
        var_features = ["var_"+i for i in features]
        dataset_other = dataset.copy().sort_values(by=["customer_id", "period"])

        for var, feat in zip(var_features, features):
            
            if mode == "numerico":
                dataset_other[var] = (
                    dataset_other[feat] - dataset_other.groupby("customer_id")[feat].shift(1)
                )
            elif mode == "porcentage":
                dataset_other[var] = (
                    dataset_other.groupby("customer_id")[feat].pct_change()
                )

        print(f"Features Creados: {var_features}")
        return dataset_other
    
    def woe(self, features_numericas: list, features_categoricas: list, dataset: pd.DataFrame, target: pd.Series):
        """Retorna una lista de Dataframes para analizar la monotonicidad de cada feature"""
        lista_df_numericos = []
        lista_df_categoricas = []
        dataset_merge = dataset.copy()
        dataset_merge["target"] = target
        epsilon = 1e-6
        
        for i in features_numericas:
            cortes = pd.qcut(dataset_merge[i], q= 5, retbins=True, duplicates= "drop")[1].tolist()
            #intervalos = pd.qcut(dataset_merge[i], q= 5).cat.codes
            dataset_merge["bin"] = pd.qcut(dataset_merge[i], q= 5, duplicates= "drop")
            
            agg = dataset_merge.groupby(
                ["bin"], observed= False
            ).agg(
                total = ("target", "count"),
                eventos = ("target", "sum")
            )
            agg["no_eventos"] = agg["total"] - agg["eventos"]
            agg["event_rate"] = agg["eventos"] / agg["total"]
            agg["pct_obs"] = agg["total"] / agg["total"].sum()
            agg["dist_eventos"] = agg["eventos"] / agg["eventos"].sum()
            agg["dist_no_eventos"] = agg["no_eventos"] / agg["no_eventos"].sum()
            agg["woe"] = np.log(
                (agg["dist_no_eventos"] + epsilon)/(agg["dist_eventos"] + epsilon)
            )
            agg["iv"] = (agg["dist_no_eventos"] - agg["dist_eventos"]) * agg["woe"]

            agg = agg.reset_index()

            feature_num = pd.DataFrame({
                "feature": i,
                "bin": range(len(agg)),
                "intervalo": agg["bin"].astype("str"),
                "regla": cortes[1:],
                "n_observaciones": agg["total"],
                "pct_obs": agg["pct_obs"],
                "n_eventos": agg["eventos"],
                "n_no_eventos": agg["no_eventos"],
                "bad_rate": agg["event_rate"],
                "dist_eventos": agg["dist_eventos"],
                "dist_no_eventos": agg["dist_no_eventos"],
                "WOE":  agg["woe"],
                "IV": agg["iv"]
            })
            
            lista_df_numericos.append(feature_num)

        for i in features_categoricas:

            agg = dataset_merge.groupby(
                [i]
            ).agg(
                total = ("target", "count"),
                eventos = ("target", "sum")
            )

            agg["no_eventos"] = agg["total"] - agg["eventos"]
            agg["event_rate"] = agg["eventos"] / agg["total"]
            agg["dist_eventos"] = agg["eventos"] / agg["eventos"].sum()
            agg["dist_no_eventos"] = agg["no_eventos"] / agg["no_eventos"].sum()
            agg["woe"] = np.log(
                (agg["dist_no_eventos"] + epsilon)/(agg["dist_eventos"] + epsilon)
            )
            agg["iv"] = (agg["dist_no_eventos"] - agg["dist_eventos"]) * agg["woe"]
            agg["pct_obs"] = agg["total"] / agg["total"].sum()

            agg = agg.reset_index()

            feature_cat = pd.DataFrame({
                "feature": i,
                "bin": range(len(agg)),
                "intervalo": agg[i].astype("str"),
                "n_observaciones": agg["total"],
                "pct_obs": agg["pct_obs"],
                "n_eventos": agg["eventos"],
                "n_no_eventos": agg["no_eventos"],
                "bad_rate": agg["event_rate"],
                "dist_eventos": agg["dist_eventos"],
                "dist_no_eventos": agg["dist_no_eventos"],
                "WOE":  agg["woe"],
                "IV": agg["iv"]
            })
            lista_df_categoricas.append(feature_cat)

        print("Lista de Dataframes creados exitosamente...")
        return lista_df_numericos, lista_df_categoricas
    
    def info_binning(self, dataset: pd.DataFrame, target: pd.Series, lista_feat: list):
        """Proporciona una tabla resumen de los features respecto a su distribución y event rate
        para hacer los binning manualmente"""
        dataset_merge = dataset.copy()
        dataset_merge["target"] = target
        lista_features = []

        for i in lista_feat:
            #feat = dataset_merge[i].value_counts(normalize = True)
            #event_rate = dataset_merge.groupby(i)["target"].mean()

            table = dataset_merge.groupby(
                i
            ).agg(
                total_obs = ("target", "count"),
                total_event = ("target", "sum"),
                event_rate = ("target", "mean")
            )
            table["pct_dist"] = table["total_obs"] / table["total_obs"].sum()

            #table = pd.concat([df_feat, event_rate], axis= 1)
            table = table.reset_index()
            table = table.sort_values(by=i, ascending= True)

            lista_features.append(table)

        return lista_features
    
    def info_binning_month_over_month(self, dataset: pd.DataFrame, target: pd.Series, lista_feat: list):
        """Proporciona una tabla resumen de los features respecto a su distribución y event rate
        para hacer los binning manualmente (Ahora viendo month over month)"""
        dataset_merge = dataset.copy()
        dataset_merge["target"] = target
        lista_features = []

        for i in lista_feat:
            #feat = dataset_merge[i].value_counts(normalize = True)
            #event_rate = dataset_merge.groupby(i)["target"].mean()

            table = dataset_merge.groupby(
                i
            ).agg(
                total_obs = ("target", "count"),
                total_event = ("target", "sum"),
                event_rate = ("target", "mean")
            )
            table["pct_dist"] = table["total_obs"] / table["total_obs"].sum()

            #table = pd.concat([df_feat, event_rate], axis= 1)
            table = table.reset_index()
            table = table.sort_values(by=i, ascending= True)

            lista_features.append(table)

        return lista_features
    
    def cut_bins(self, dataset: pd.DataFrame, intervalos: list, labels: list, feature: str):
        """Crea los bins mediante pd.cut y devuelve un dataframe con las nuevas distribuciones y event rate"""
        dataset_merge = dataset.copy()
        
        intervalos_formato = intervalos[1:]
        epsilon = 1e-6

        dataset_merge["bins"] = pd.cut(
            dataset_merge[feature],
            bins= intervalos,
            labels= labels
        )

        agg = dataset_merge.groupby(
            "bins"
        ).agg(
            n_observaciones = ("total_obs", "sum"),
            n_eventos = ("total_event", "sum"),
        )
        
        agg["bad_rate"] = agg["n_eventos"] / agg["n_observaciones"]
        agg["pct_obs"] = agg["n_observaciones"] / agg["n_observaciones"].sum()
        agg["n_no_eventos"] = agg["n_observaciones"] - agg["n_eventos"]
        agg["dist_eventos"] = agg["n_eventos"] / agg["n_eventos"].sum()
        agg["dist_no_eventos"] = agg["n_no_eventos"] / agg["n_no_eventos"].sum()
        agg["woe"] = np.log(
            (agg["dist_no_eventos"] + epsilon) / (agg["dist_eventos"] + epsilon)
        )
        agg["iv"] = (agg["dist_no_eventos"] - agg["dist_eventos"]) * agg["woe"]

        table = pd.DataFrame({
            "feature": feature,
            "bin": range(len(agg)),
            "regla": intervalos_formato,
            "n_observaciones": agg["n_observaciones"],
            "pct_obs": agg["pct_obs"],
            "n_eventos": agg["n_eventos"],
            "n_no_eventos": agg["n_no_eventos"],
            "bad_rate": agg["bad_rate"],
            "dist_eventos": agg["dist_eventos"],
            "dist_no_eventos": agg["dist_no_eventos"],
            "WOE":  agg["woe"],
            "IV": agg["iv"]
        })

        print(f"IV del feature: {round(agg['iv'].sum(), 4)}")
        return table
    
    def cut_bins_month_over_moth(self, dataset: pd.DataFrame, intervalos: list, labels: list, feature: str, lista_periodos_base: list, lista_periodos_comparacion: list, target_base: pd.Series):
        """Crea los bins mediante pd.cut y devuelve un dataframe con las nuevas distribuciones y event rate"""
        dataset_merge = dataset.copy()
        target_baseline = target_base.copy()
        
        intervalos_formato = intervalos[1:]
        epsilon = 1e-6

        for a, b in zip(lista_periodos_base, lista_periodos_comparacion):

            a = pd.Timestamp(a)
            b = pd.Timestamp(b)

            dataset_merge_a = dataset_merge[dataset_merge["period"] == a]
            target_base_vf = target_baseline[target_baseline["period"] == a]

            dataset_merge_a["bins"] = pd.cut(
                dataset_merge_a[feature],
                bins= intervalos,
                labels= labels
            )

            agg = dataset_merge_a.groupby(
                "bins"
            ).agg(
                n_observaciones = ("total_obs", "sum"),
                n_eventos = ("total_event", "sum"),
            )
            
            agg["bad_rate"] = agg["n_eventos"] / agg["n_observaciones"]
            agg["pct_obs"] = agg["n_observaciones"] / agg["n_observaciones"].sum()
            agg["n_no_eventos"] = agg["n_observaciones"] - agg["n_eventos"]
            agg["dist_eventos"] = agg["n_eventos"] / agg["n_eventos"].sum()
            agg["dist_no_eventos"] = agg["n_no_eventos"] / agg["n_no_eventos"].sum()
            agg["woe"] = np.log(
                (agg["dist_no_eventos"] + epsilon) / (agg["dist_eventos"] + epsilon)
            )
            agg["iv"] = (agg["dist_no_eventos"] - agg["dist_eventos"]) * agg["woe"]

            table = pd.DataFrame({
                "feature": feature,
                "bin": range(len(agg)),
                "regla": intervalos_formato,
                "n_observaciones": agg["n_observaciones"],
                "pct_obs": agg["pct_obs"],
                "n_eventos": agg["n_eventos"],
                "n_no_eventos": agg["n_no_eventos"],
                "bad_rate": agg["bad_rate"],
                "dist_eventos": agg["dist_eventos"],
                "dist_no_eventos": agg["dist_no_eventos"],
                "WOE":  agg["woe"],
                "IV": agg["iv"]
            })

            print(f"IV del feature: {round(agg['iv'].sum(), 4)}")
        return table

    def woe_2(self, features_numericas: list, features_categoricas: list, dataset: pd.DataFrame, target: pd.Series, mode = "baseline", labels = [], bins = [], categories_base = []):
        """Retorna una lista de Dataframes para analizar la monotonicidad de cada feature"""
        lista_df_numericos = []
        lista_df_categoricas = []
        dataset_merge = dataset.copy()
        dataset_merge["target"] = target
        epsilon = 1e-6
        
        for x, i in enumerate(features_numericas):
            #print(x, i)
            
            if mode == "baseline":
                cortes = pd.qcut(dataset_merge[i], q= 5, retbins=True, duplicates= "drop")[1].tolist()
                #intervalos = pd.qcut(dataset_merge[i], q= 5).cat.codes
                dataset_merge["bin"] = pd.qcut(dataset_merge[i], q= 5, duplicates= "drop")

            elif mode == "future":
                cortes = bins[x]
                #cortes = cortes[1:]
                #print(cortes)
                #print(type(cortes))
                dataset_merge["bin"] = pd.cut(
                    dataset_merge[i],
                    labels= labels[x],
                    bins= bins[x]
                )
                #print(f"Tamaño dataset: {len(dataset_merge)}")
            
            agg = dataset_merge.groupby(
                ["bin"], observed= False
            ).agg(
                total = ("target", "count"),
                eventos = ("target", "sum")
            )
            #print(f"Tamaño agg: {len(agg)}")
            agg["no_eventos"] = agg["total"] - agg["eventos"]
            agg["event_rate"] = agg["eventos"] / agg["total"]
            agg["pct_obs"] = agg["total"] / agg["total"].sum()
            agg["dist_eventos"] = agg["eventos"] / agg["eventos"].sum()
            agg["dist_no_eventos"] = agg["no_eventos"] / agg["no_eventos"].sum()
            agg["woe"] = np.log(
                (agg["dist_no_eventos"] + epsilon)/(agg["dist_eventos"] + epsilon)
            )
            agg["iv"] = (agg["dist_no_eventos"] - agg["dist_eventos"]) * agg["woe"]

            agg = agg.reset_index()
            #print(len(agg))

            feature_num = pd.DataFrame({
                "feature": i,
                "bin": range(len(agg)),
                "intervalo": agg["bin"].astype("str"),
                "regla": cortes[1:],
                "n_observaciones": agg["total"],
                "pct_obs": agg["pct_obs"],
                "n_eventos": agg["eventos"],
                "n_no_eventos": agg["no_eventos"],
                "bad_rate": agg["event_rate"],
                "dist_eventos": agg["dist_eventos"],
                "dist_no_eventos": agg["dist_no_eventos"],
                "WOE":  agg["woe"],
                "IV": agg["iv"]
            })
            
            lista_df_numericos.append(feature_num)

        for x, i in enumerate(features_categoricas):

            if mode == "baseline":

                agg = dataset_merge.groupby(
                    [i]
                ).agg(
                    total = ("target", "count"),
                    eventos = ("target", "sum")
                )
            
            elif mode == "future":

                cats = categories_base[x]
                dataset_merge[i] = dataset_merge[i].where(
                    dataset_merge[i].isin(cats),
                    "OTHER"
                )

                agg = dataset_merge.groupby(
                    [i]
                ).agg(
                    total = ("target", "count"),
                    eventos = ("target", "sum")
                )
                agg = agg.reindex(cats + ["OTHER"], fill_value= 0)

            
            agg["no_eventos"] = agg["total"] - agg["eventos"]
            agg["event_rate"] = agg["eventos"] / agg["total"]
            agg["dist_eventos"] = agg["eventos"] / agg["eventos"].sum()
            agg["dist_no_eventos"] = agg["no_eventos"] / agg["no_eventos"].sum()
            agg["woe"] = np.log(
                (agg["dist_no_eventos"] + epsilon)/(agg["dist_eventos"] + epsilon)
            )
            agg["iv"] = (agg["dist_no_eventos"] - agg["dist_eventos"]) * agg["woe"]
            agg["pct_obs"] = agg["total"] / agg["total"].sum()

            agg = agg.reset_index()

            feature_cat = pd.DataFrame({
                "feature": i,
                "bin": range(len(agg)),
                "intervalo": agg[i].astype("str"),
                "n_observaciones": agg["total"],
                "pct_obs": agg["pct_obs"],
                "n_eventos": agg["eventos"],
                "n_no_eventos": agg["no_eventos"],
                "bad_rate": agg["event_rate"],
                "dist_eventos": agg["dist_eventos"],
                "dist_no_eventos": agg["dist_no_eventos"],
                "WOE":  agg["woe"],
                "IV": agg["iv"]
            })
            lista_df_categoricas.append(feature_cat)

        print("Lista de Dataframes creados exitosamente...")
        return lista_df_numericos, lista_df_categoricas
    
    def _pre_bins(self, bins: list):
        """Pre procesa los bins numéricos incluyendo los rangos -np.inf y np.inf"""
        bins[-1] = np.inf
        bins.insert(0, -np.inf)
        return bins
    
    def _pre_psi(self, dataset_num: pd.DataFrame, dataset_cat: pd.DataFrame, dataset_comparative: pd.DataFrame, target_comparative: pd.Series):
        """Pre procesa la lista de dataframe para obtener los labels y bins"""

        ## Numéricas
        lista_feat_num = []
        list_reglas_num = []
        list_labels_num = []

        print("--Ordenando los dataset numéricos baseline--")
        for i in dataset_num:
            feat = str.join('',i['feature'].unique().tolist())
            regla = [d for d in i["regla"].to_list()]
            #print("-------Empezamiento proceso pre-bins-------")
            regla = self._pre_bins(regla)
            labels = [d for d in i["bin"].to_list()]
            lista_feat_num.append(feat)
            list_reglas_num.append(regla)
            list_labels_num.append(labels)

        lista_feat_cat = []
        list_categorias_cat = []

        ## Categóricas
        print("--Ordenando los dataset categóricos baseline--")
        for i in dataset_cat:
            feat = str.join('',i['feature'].unique().tolist())
            cats = [d for d in i["intervalo"].to_list()]
            labels = [d for d in i["bin"].to_list()]
            lista_feat_cat.append(feat)
            list_categorias_cat.append(cats)

        ## Aplicar el woe_2 al dataset comparativo (test/OOT)
        print("--Aplicando el WOE para los datos de comparación--")
        df_num_vf, df_cat_vf = self.woe_2(
            features_numericas =  lista_feat_num,
            features_categoricas = lista_feat_cat,
            dataset = dataset_comparative,
            target = target_comparative,
            mode = "future",
            labels = list_labels_num,
            bins = list_reglas_num,
            categories_base = list_categorias_cat
        )
        
        return df_num_vf, df_cat_vf


    def psi(self, dataset_base_manual: list, dataset_base: pd.DataFrame, database_comparative: pd.DataFrame, feature_num_auto: list, feature_cat: list, target_base: pd.Series, target_comparative: pd.Series):
        """Pipeline completo para calcular el PSI tanto del base como del comparative"""
        
        dataset_base_auto = dataset_base.copy()
        database_comparative_vf = database_comparative.copy()
        eps = 1e-6
        
        print("Aplicando el método WOE para el baseline...")
        df_num, df_cat = self.woe(features_categoricas = feature_cat, features_numericas= feature_num_auto, target= target_base, dataset= dataset_base_auto)
        
        print("Concatenando los datasets manuales a la lista numérica automática...")
        for i in dataset_base_manual:
            df_num.append(i)
        
        print("Aplicando el método WOE para la comparación...")
        df_num_comp, df_cat_comp = self._pre_psi(dataset_num= df_num, dataset_cat = df_cat, dataset_comparative= database_comparative_vf, target_comparative = target_comparative)

        list_num_psi = []
        list_cat_psi = []

        list_feat_num = []
        list_psi_num = []
        list_iv_base_num = []
        list_iv_comp_num = []

        print("Iniciando el cruce entre baseline y comparación para numéricas...")
        for base, comp in zip(df_num, df_num_comp):
            df_tmp = comp.merge(base, how = "left", on = ["feature" , "bin"], suffixes = ("", "_base"))
            feat_num = ["feature", "bin", "pct_obs", "pct_obs_base", "WOE", "WOE_base", "IV", "IV_base"]
            df_tmp = df_tmp[feat_num]
            df_tmp["psi"] = ( df_tmp["pct_obs_base"].clip(eps, 1) - df_tmp["pct_obs"].clip(eps, 1) ) * np.log( (df_tmp["pct_obs_base"].clip(eps, 1)) / (df_tmp["pct_obs"].clip(eps, 1))  )
            
            feature = df_tmp["feature"].unique()[0]
            psi = df_tmp["psi"].sum()
            iv_base = df_tmp["IV_base"].sum()
            iv_comp = df_tmp["IV"].sum()

            list_num_psi.append(df_tmp)

            list_feat_num.append(feature)
            list_psi_num.append(psi)
            list_iv_base_num.append(iv_base)
            list_iv_comp_num.append(iv_comp)

        print("Iniciando el cruce entre baseline y comparación para categóricas...")
        for base, comp in zip(df_cat, df_cat_comp):
            df_tmp = comp.merge(base, how = "left", on = ["feature" , "intervalo"], suffixes = ("", "_base"))
            feat_num = ["feature", "bin", "bin_base", "intervalo", "pct_obs", "pct_obs_base", "WOE", "WOE_base", "IV", "IV_base"]
            df_tmp = df_tmp[feat_num]
            df_tmp["psi"] = ( df_tmp["pct_obs_base"].clip(eps, 1) - df_tmp["pct_obs"].clip(eps, 1) ) * np.log( (df_tmp["pct_obs_base"].clip(eps, 1)) / (df_tmp["pct_obs"].clip(eps, 1))  )
            
            feature = df_tmp["feature"].unique()[0]
            psi = df_tmp["psi"].sum()
            iv_base = df_tmp["IV_base"].sum()
            iv_comp = df_tmp["IV"].sum()

            list_num_psi.append(df_tmp)

            list_feat_num.append(feature)
            list_psi_num.append(psi)
            list_iv_base_num.append(iv_base)
            list_iv_comp_num.append(iv_comp)

        print("Creando el resumen...")

        df_overall = pd.DataFrame({
            "Feature": list_feat_num,
            "PSI": list_psi_num,
            "IV Base": list_iv_base_num,
            "IV Comparative": list_iv_comp_num
        }).sort_values(by= "PSI", ascending= False)

        return df_overall, list_num_psi
        
    def psi_month_over_month(self, dataset_base_manual: list, dataset_base: pd.DataFrame, database_comparative: pd.DataFrame, feature_num_auto: list, feature_cat: list, target_base: pd.Series, target_comparative: pd.Series, lista_periodos_base: list, lista_periodos_comparacion: list):
        """Pipeline completo para calcular el PSI tanto del base como del comparative"""
        
        dataset_base_auto_vf = dataset_base.copy()
        dataset_base_comparative_vf = dataset_base.copy()
        
        database_comparative_vf = database_comparative.copy()
        target_base_vf = target_base.copy()
        target_comparative_vf = target_comparative.copy()
    
        eps = 1e-6

        list_num_psi = []
        list_cat_psi = []

        list_feat_num = []
        list_psi_num = []
        list_iv_base_num = []
        list_iv_comp_num = []

        for z, (a, b) in enumerate(zip(lista_periodos_base, lista_periodos_comparacion)):
            a = pd.Timestamp(a)
            b = pd.Timestamp(b)
            dataset_base_auto = dataset_base_auto_vf[dataset_base_auto_vf["period"] == a]
            dataset_base_comparative = dataset_base_comparative_vf[dataset_base_comparative_vf["period"] == b]
            target_base_i = target_base_vf[target_base_vf["period"] == a]
            target_comparative_i = target_base_vf[target_base_vf["period"] == b]

            print("Aplicando el método WOE para el baseline...")
            df_num, df_cat = self.woe(features_categoricas = feature_cat, features_numericas= feature_num_auto, target= target_base_i, dataset= dataset_base_auto)
            
            print("Concatenando los datasets manuales a la lista numérica automática...")
            df_num = df_num + dataset_base_manual[z]
            
            print("Aplicando el método WOE para la comparación...")
            df_num_comp, df_cat_comp = self._pre_psi(dataset_num= df_num, dataset_cat = df_cat, dataset_comparative= dataset_base_comparative, target_comparative = target_comparative_i)

            print("Iniciando el cruce entre baseline y comparación para numéricas...")
            for base, comp in zip(df_num, df_num_comp):
                df_tmp = comp.merge(base, how = "left", on = ["feature" , "bin"], suffixes = ("", "_base"))
                feat_num = ["feature", "bin", "pct_obs", "pct_obs_base", "WOE", "WOE_base", "IV", "IV_base"]
                df_tmp = df_tmp[feat_num]
                df_tmp["psi"] = ( df_tmp["pct_obs_base"].clip(eps, 1) - df_tmp["pct_obs"].clip(eps, 1) ) * np.log( (df_tmp["pct_obs_base"].clip(eps, 1)) / (df_tmp["pct_obs"].clip(eps, 1))  )
                
                feature = df_tmp["feature"].unique()[0]
                psi = df_tmp["psi"].sum()
                iv_base = df_tmp["IV_base"].sum()
                iv_comp = df_tmp["IV"].sum()

                list_num_psi.append(df_tmp)

                list_feat_num.append(feature)
                list_psi_num.append(psi)
                list_iv_base_num.append(iv_base)
                list_iv_comp_num.append(iv_comp)

            print("Iniciando el cruce entre baseline y comparación para categóricas...")
            for base, comp in zip(df_cat, df_cat_comp):
                df_tmp = comp.merge(base, how = "left", on = ["feature" , "intervalo"], suffixes = ("", "_base"))
                feat_num = ["feature", "bin", "bin_base", "intervalo", "pct_obs", "pct_obs_base", "WOE", "WOE_base", "IV", "IV_base"]
                df_tmp = df_tmp[feat_num]
                df_tmp["psi"] = ( df_tmp["pct_obs_base"].clip(eps, 1) - df_tmp["pct_obs"].clip(eps, 1) ) * np.log( (df_tmp["pct_obs_base"].clip(eps, 1)) / (df_tmp["pct_obs"].clip(eps, 1))  )
                
                feature = df_tmp["feature"].unique()[0]
                psi = df_tmp["psi"].sum()
                iv_base = df_tmp["IV_base"].sum()
                iv_comp = df_tmp["IV"].sum()

                list_num_psi.append(df_tmp)

                list_feat_num.append(feature)
                list_psi_num.append(psi)
                list_iv_base_num.append(iv_base)
                list_iv_comp_num.append(iv_comp)

            print("Creando el resumen...")
            etiqueta = f"{b} vs {a}"

            df_overall = pd.DataFrame({
                "Feature": list_feat_num,
                "Periodo": etiqueta,
                "PSI": list_psi_num,
                "IV Base": list_iv_base_num,
                "IV Comparative": list_iv_comp_num
            }).sort_values(by= "PSI", ascending= False)

        return df_overall, list_num_psi        
        

        





class Modeling:
    def __init__(self):
        pass

    def objective(self, trial, X_train, y_train):
        params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "random_state": 42,
        "n_jobs": -1
    }

        model = RandomForestClassifier(**params)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        score = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",  
            n_jobs=-1
        ).mean()

        return score
    
    def objective_xgboost(self, trial, X_train, y_train, inicio):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "base_score": inicio
        }

        model = XGBClassifier(**params)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        score = cross_val_score(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        ).mean()

        return score

    
    def objective_lightlgb(self, trial, X_train, y_train):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1
        }

        model = LGBMClassifier(**params)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        score = cross_val_score(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        ).mean()

        return score
    

class Explicabilidad():
    def __init__(self):
        pass

    def shap_binary_class1(self, model, X_sample):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)

        if len(shap_values.values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        return shap_values





















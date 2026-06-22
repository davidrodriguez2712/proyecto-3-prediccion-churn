import pandas as pd
import numpy as np


class EDA:
    def __init__(self):
        pass
    
    def information_value(self, dataframe: pd.DataFrame, features: list, target_serie: pd.Series, target_name):
        """Retorna una tabla resumen del IV de cada feature numérica"""
        df = dataframe.copy()
        target = target_serie.copy()
        df_merge = pd.concat([df[features], target_serie], axis = 1)
        lista_iv = []
        lista_bins = []

        for i in features:
            df_merge['bins'], bins = pd.qcut(
                x = df[i],
                q = 5,
                retbins= True,
                duplicates= 'drop'
            )
            eps = 1e-5
            table = df_merge.groupby(
                ['bins'], observed=False
            ).agg(
                total_obs = (target_name, "count"),
                total_event = (target_name, "sum"),
                event_rate = (target_name, "mean")
            )
            table['total_no_event'] = table["total_obs"] - table["total_event"]
            table['no_event_rate'] = table['total_no_event'] / table['total_obs']
            table['dist_event'] = table['total_event'] / table['total_event'].sum()
            table['dist_no_event'] = table['total_no_event'] / table['total_no_event'].sum()
            table['woe'] = (
                np.log(
                    (table['dist_no_event'] + eps) / (table['dist_event'] + eps)
                )
            )
            table = table.reset_index(level= 0, drop= False)
            table['information_value_calculation'] = ((table['dist_no_event'] - table['dist_event']) * (table['woe']))
            iv = (table['information_value_calculation'].sum())
            #print(table['woe'].count())
            lista_iv.append(iv)
            lista_bins.append(bins)
        
        df_iv = pd.DataFrame({
            'Feature': features,
            'Information Value': lista_iv
        }).sort_values(by= 'Information Value', ascending= False)

        return df_iv, lista_bins



    def binning_explore(self, dataset: pd.DataFrame, target: pd.Series, lista_feat: list):
        """Proporciona una tabla resumen de los features respecto a su distribución y event rate
        para hacer los binning manualmente"""
        dataset_merge = dataset.copy()
        dataset_merge["target"] = target
        lista_features = []

        for i in lista_feat:
            #feat = dataset_merge[i].value_counts(normalize = True)
            #event_rate = dataset_merge.groupby(i)["target"].mean()
            dataset_merge['bins'] = pd.qcut(
                x = dataset_merge[i],
                q = 5
            )
            eps = 1e-5
            table = dataset_merge.groupby(
                ['bins'], observed=False
            ).agg(
                total_obs = ("target", "count"),
                total_event = ("target", "sum"),
                event_rate = ("target", "mean")
            )
            table["pct_dist"] = table["total_obs"] / table["total_obs"].sum()
            table["total_no_event"] = table["total_obs"] - table["total_event"]
            table["dist_eventos"] = table["total_event"] / table["total_event"].sum()
            table["dist_no_eventos"] = table["total_no_event"] / table["total_no_event"].sum()
            table["woe"] = (
                np.log( (table["dist_no_eventos"] + eps) / (table["dist_eventos"] + eps) )
            )
            table["feature"] = i

            #table = pd.concat([df_feat, event_rate], axis= 1)
            table = table.reset_index()
            #table = table.sort_values(by=i, ascending= True)
            cols = ['feature'] + [col for col in table.columns if col != 'feature']
            table = table[cols]

            lista_features.append(table)

        return lista_features
    
    def bins_cut(self, dataset: pd.DataFrame, intervalos: list, labels: list, feature: str, target: str):
        """Crea los bins mediante pd.cut y devuelve un dataframe con las nuevas distribuciones y event rate"""
        dataset_merge = dataset.copy()
        
        intervalos_formato = intervalos[1:]
        epsilon = 1e-5

        dataset_merge["bins"] = pd.cut(
            dataset_merge[feature],
            bins= intervalos,
            labels= labels
        )

        agg = dataset_merge.groupby(
            "bins", observed=False
        ).agg(
            n_observaciones = (target, "count"),
            n_eventos = (target, "sum"),
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
    
    def binning_table(self, dataset: pd.DataFrame, target: pd.Series, feature_bin: str):
        """Proporciona una tabla resumen de los features respecto a su distribución y event rate"""
        dataset_merge = dataset.copy()
        dataset_merge["target"] = target

        eps = 1e-5
        table = dataset_merge.groupby(
            [feature_bin], observed=False
        ).agg(
            total_obs = ("target", "count"),
            total_event = ("target", "sum"),
            event_rate = ("target", "mean")
        )
        table["pct_dist"] = table["total_obs"] / table["total_obs"].sum()
        table["total_no_event"] = table["total_obs"] - table["total_event"]
        table["dist_eventos"] = table["total_event"] / table["total_event"].sum()
        table["dist_no_eventos"] = table["total_no_event"] / table["total_no_event"].sum()
        table["woe"] = (
            np.log( (table["dist_no_eventos"] + eps) / (table["dist_eventos"] + eps) )
        )
        table["feature"] = feature_bin

        #table = pd.concat([df_feat, event_rate], axis= 1)
        table = table.reset_index()
        #table = table.sort_values(by=i, ascending= True)
        cols = ['feature'] + [col for col in table.columns if col != 'feature']
        table = table[cols]

        return table









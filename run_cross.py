# ## Imports
import pandas as pd
import numpy as np

#Data preparation
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE

#TCA+
from tl_algs import tca_plus

#Old REPD model
from REPD_Impl import REPD
from autoencoder import AutoEncoder

#Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#Performance metrics
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import sklearn.metrics as metrics
from sklearn.metrics import f1_score

#Result presentation
from tabulate import tabulate

#Visualization
from matplotlib import pyplot as plt

#Other
import warnings


# ## Load Dataset support

def load_arff(dataset_name, data_preparation_function):
    #Load data
    data, _ = arff.loadarff("./data/"+dataset+".arff")
    
     # Wrap data into a pandas dataframe
    df = pd.DataFrame(data)
    
    #Prepare data
    df = data_preparation_function(df)
    
    #Return dataframe
    return df

def load_csv(dataset_name, data_preparation_function):
    #Load data
    data = pd.read_csv("./data/"+dataset_name+".csv")
    
    #Prepare data
    df = data_preparation_function(data)
    
    #Return dataframe
    return df

def load_data(dataset_name, dataset_settings):
    if dataset_settings["type"] == "arff":
        return load_arff(dataset_name, dataset_settings["prep_func"])
    elif dataset_settings["type"] == "csv":
        return load_csv(dataset_name, dataset_settings["prep_func"])
    pass

def get_defect_0_1_prep_func(defect_column_name, mapping_function):
    #
    def defect_0_1_prep_func(df):
        df[defect_column_name] = df[defect_column_name].map(mapping_function)
        return df
    #
    return defect_0_1_prep_func

def get_cleanup_prep_func_decorator(prep_func):
    #
    def decorated_prep_func_1(df):
        df = prep_func(df)
        
        #Remove all with missing values
        df = df.dropna()

        #Remove duplicate instances
        df = df.drop_duplicates()
        #
        return df
    #
    return decorated_prep_func_1


def get_adjust_defect_column_name_prep_func_decorator(defecive_column_name, prep_func):
    #
    def decorated_prep_func_2(df):
        df = prep_func(df)
        
        #Rename column
        df = df.rename(columns={defecive_column_name: "defective"})
        #
        return df
    #
    return decorated_prep_func_2        

def get_remove_column_prep_func_decorator(columns_to_remove, prep_func):
    def decorated_prep_func_3(df):
        df = prep_func(df)
        
        #Drop columns
        df = df.drop(columns=columns_to_remove)
        #
        return df
    #
    return decorated_prep_func_3


# ## Model

class REPD_EX:
    
    def __init__(self, base_repd, defective_classification_model, non_defective_classification_model, use_def_m=True, use_non_def_m=True):
        self.base_repd = base_repd
        self.defective_classification_model = defective_classification_model
        self.non_defective_classification_model = non_defective_classification_model
        self.use_def_m=use_def_m
        self.use_non_def_m=use_non_def_m
    
    def fit(self, X, y, train_base=True):
        #
        if train_base:
            X_m1_fit, X_m2_fit, y_m1_fit, y_m2_fit = train_test_split(X, y, test_size=0.5)
            #
            oversample = SMOTE()
            X_m1_fit, y_m1_fit = oversample.fit_resample(X_m1_fit, y_m1_fit)
            #
            self.base_repd.fit(X_m1_fit,y_m1_fit)
        else:
            X_m2_fit = X
            y_m2_fit = y
        #
        y_p = self.base_repd.predict(X_m2_fit)
        X_r = pd.DataFrame(self.base_repd.transform(X_m2_fit))
        #
        #========================================
        if self.use_def_m:
            X_s_o_t = X_m2_fit[y_p==1]
            X_s_r_t = X_r[y_p==1]
            #
            X_s_o_t = pd.DataFrame(X_s_o_t.values)
            X_s_r_t = pd.DataFrame(X_s_r_t.values)
            #
            X_s_t = pd.concat([X_s_o_t, X_s_r_t], axis=1, join="inner")
            #
            y_s_t = y_m2_fit[y_p==1]
            #
            #print("P Defective:",len(y_s_t[y_s_t==1]),"/",len(y_s_t))
            #
            oversample = SMOTE()
            X_s_t, y_s_t = oversample.fit_resample(X_s_t, y_s_t)
            #
            self.defective_classification_model.fit(X_s_t, y_s_t)
        #========================================
        if self.use_non_def_m:
            X_s_o_t = X_m2_fit[y_p==0]
            X_s_r_t = X_r[y_p==0]
            #
            X_s_o_t = pd.DataFrame(X_s_o_t.values)
            X_s_r_t = pd.DataFrame(X_s_r_t.values)
            #
            X_s_t = pd.concat([X_s_o_t, X_s_r_t], axis=1, join="inner")
            #
            y_s_t = y_m2_fit[y_p==0]
            #
            #print("P Defective:",len(y_s_t[y_s_t==1]),"/",len(y_s_t))
            #
            oversample = SMOTE()
            X_s_t, y_s_t = oversample.fit_resample(X_s_t, y_s_t)
            #
            self.non_defective_classification_model.fit(X_s_t, y_s_t)
        #========================================
        return self
    
    def predict(self, X):
        y_p = self.base_repd.predict(X)
        #
        X_r = pd.DataFrame(self.base_repd.transform(X))
        #
        #========================================
        if self.use_def_m:
            X_s_o_t = X[y_p==1]
            X_s_r_t = X_r[y_p==1]
            #
            X_s_o_t = pd.DataFrame(X_s_o_t.values)
            X_s_r_t = pd.DataFrame(X_s_r_t.values)
            #
            X_s_t = pd.concat([X_s_o_t, X_s_r_t], axis=1, join="inner")
            #
            y_s_p_p = self.defective_classification_model.predict(X_s_t)
        #========================================
        if self.use_non_def_m:
            X_s_o_t = X[y_p==0]
            X_s_r_t = X_r[y_p==0]
            #
            X_s_o_t = pd.DataFrame(X_s_o_t.values)
            X_s_r_t = pd.DataFrame(X_s_r_t.values)
            #
            X_s_t = pd.concat([X_s_o_t, X_s_r_t], axis=1, join="inner")
            #
            y_s_n_p = self.non_defective_classification_model.predict(X_s_t)
        #========================================
        r = []
        #
        cnt1 = 0
        k1 = 0
        cnt2 = 0
        k2 = 0
        for v in y_p:
            if v == 0:
                if self.use_non_def_m:
                    if hasattr(y_s_n_p[k2], "__len__"):
                        nmr = y_s_n_p[k2][0]
                    else:
                        nmr = y_s_n_p[k2]
                    if nmr >= 0.5:
                        r.append(1)
                    else:
                        r.append(0)
                        cnt2 = cnt2 + 1
                    k2 = k2 + 1
                else:
                    r.append(0)
            else:
                if self.use_def_m:
                    if hasattr(y_s_p_p[k1], "__len__"):
                        nmr = y_s_p_p[k1][0]
                    else:
                        nmr = y_s_p_p[k1]
                    if nmr >= 0.5:
                        r.append(1)
                        cnt1 = cnt1 + 1
                    else:
                        r.append(0)
                    k1 = k1 + 1
                else:
                    r.append(1)
        #
        #print("S:", len(X), "P:", len(y_s_p_p), "C1:", cnt1, "N:", len(y_s_n_p), "C2:", cnt2)
        #
        return y_p, np.asarray(r)


# ## Calculate performance data

def calculate_results(y_true,y_predicted):
    accuracy = accuracy_score(y_true, y_predicted)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_predicted, average='binary')
    return accuracy, precision, recall, f1_score


# ## Visualize results

def plot_res(column):
    bp_dict = results_df.boxplot(
        column=column,
        by=["Model"],
        layout=(1,1),       
        return_type='both',
        patch_artist = True,
        vert=False,
    )    
    plt.suptitle("")
    plt.show()


# ## Experiment

source_datasets = [
    "JDT_R2_0","JDT_R2_1","JDT_R3_0","JDT_R3_1","JDT_R3_2"
]
target_datasets = [
    "PDE_R2_0","PDE_R2_1","PDE_R3_0","PDE_R3_1","PDE_R3_2"    
]


# In[ ]:


dataset_settings = {
  "JDT_R2_0": {
      "type":"csv",
      "prep_func": get_cleanup_prep_func_decorator(
                      get_remove_column_prep_func_decorator( ["File"],
                      get_adjust_defect_column_name_prep_func_decorator("bug_cnt",
                          get_defect_0_1_prep_func("bug_cnt", lambda x: 1 if x>0 else 0)
                    ))
                  )
  },
  "JDT_R2_1": {
      "type":"csv",
      "prep_func": get_cleanup_prep_func_decorator(
                      get_remove_column_prep_func_decorator( ["File"],
                      get_adjust_defect_column_name_prep_func_decorator("bug_cnt",
                          get_defect_0_1_prep_func("bug_cnt", lambda x: 1 if x>0 else 0)
                    ))
                  )      
  },
  "JDT_R3_0": {
      "type":"csv",
      "prep_func": get_cleanup_prep_func_decorator(
                      get_remove_column_prep_func_decorator( ["File"],
                          get_adjust_defect_column_name_prep_func_decorator("bug_cnt",
                              get_defect_0_1_prep_func("bug_cnt", lambda x: 1 if x>0 else 0)
                      )
                    )
                  )      
  },
  "JDT_R3_1": {
      "type":"csv",
      "prep_func": get_cleanup_prep_func_decorator(
                      get_remove_column_prep_func_decorator( ["File"],
                          get_adjust_defect_column_name_prep_func_decorator("bug_cnt",
                              get_defect_0_1_prep_func("bug_cnt", lambda x: 1 if x>0 else 0)
                      )
                    )
                  )
      
  },
  "JDT_R3_2": {
      "type":"csv",
      "prep_func": get_cleanup_prep_func_decorator(
                      get_remove_column_prep_func_decorator( ["File"],
                          get_adjust_defect_column_name_prep_func_decorator("bug_cnt",
                              get_defect_0_1_prep_func("bug_cnt", lambda x: 1 if x>0 else 0)
                        )
                    )
                  )      
  },
  "PDE_R2_0": {
      "type":"csv",
      "prep_func": get_cleanup_prep_func_decorator(
                      get_remove_column_prep_func_decorator( ["File"],
                      get_adjust_defect_column_name_prep_func_decorator("bug_cnt",
                          get_defect_0_1_prep_func("bug_cnt", lambda x: 1 if x>0 else 0)
                    ))
                  )
  },
  "PDE_R2_1": {
      "type":"csv",
      "prep_func": get_cleanup_prep_func_decorator(
                      get_remove_column_prep_func_decorator( ["File"],
                      get_adjust_defect_column_name_prep_func_decorator("bug_cnt",
                          get_defect_0_1_prep_func("bug_cnt", lambda x: 1 if x>0 else 0)
                    ))
                  )      
  },
  "PDE_R3_0": {
      "type":"csv",
      "prep_func": get_cleanup_prep_func_decorator(
                      get_remove_column_prep_func_decorator( ["File"],
                          get_adjust_defect_column_name_prep_func_decorator("bug_cnt",
                              get_defect_0_1_prep_func("bug_cnt", lambda x: 1 if x>0 else 0)
                      )
                    )
                  )      
  },
  "PDE_R3_1": {
      "type":"csv",
      "prep_func": get_cleanup_prep_func_decorator(
                      get_remove_column_prep_func_decorator( ["File"],
                          get_adjust_defect_column_name_prep_func_decorator("bug_cnt",
                              get_defect_0_1_prep_func("bug_cnt", lambda x: 1 if x>0 else 0)
                      )
                    )
                  )
      
  },
  "PDE_R3_2": {
      "type":"csv",
      "prep_func": get_cleanup_prep_func_decorator(
                      get_remove_column_prep_func_decorator( ["File"],
                          get_adjust_defect_column_name_prep_func_decorator("bug_cnt",
                              get_defect_0_1_prep_func("bug_cnt", lambda x: 1 if x>0 else 0)
                        )
                    )
                  )      
  }
}


dataset_data = {
    
}


for datasets, datasets_type in [(source_datasets,"source"),(target_datasets,"target")]:
    for dataset in datasets:
        print("Loading dataset: ", dataset)
        df = load_data(dataset,dataset_settings[dataset])
        #
        dataset_data[dataset] = {
            "df": df,
            "X": df.drop(columns=["defective"]),
            "y": df["defective"],
            "share": (len(df[df["defective"]==1])/ len(df)),
            "type": datasets_type
        }
        #
        print("Defective:", len(df[df["defective"]==1]), "/", len(df),"=",(len(df[df["defective"]==1])/ len(df)))
        print()


REPETITION_COUNT = 30
TEST_SIZE = 0.5


for dataset in dataset_data:
    dataset_data[dataset]["train"] = []
    dataset_data[dataset]["test"] = []
    # 
    X = dataset_data[dataset]["X"]
    y = dataset_data[dataset]["y"]
    #
    for _ in range(REPETITION_COUNT):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
        #
        dataset_data[dataset]["train"].append((X_train, y_train))
        dataset_data[dataset]["test"].append((X_test, y_test))


def normalize_train(X_train):
    train_min = X_train.min()
    train_dim = X_train.max() - X_train.min()
    train_dim[train_dim == 0] = 1
    #
    X_train = (X_train - train_min) / train_dim
    #
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    #
    return X_train, train_min, train_dim

def normalize_test(X_test, train_min, train_dim):
    X_test = (X_test - train_min) / train_dim
    #
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    #
    return X_test


warnings.simplefilter("ignore")

for mode, cls_w_c1, cls_w_c2, use_c1, use_c2 in [
                                ("balance",{0:1., 1: 2.},{0:2., 1: 1.},True, True), 
                                ("hp",{0:1.2, 1: 1.},{0:1., 1: 1.},True,False)
                                ]:
    for source in dataset_data:
        #
        if dataset_data[source]["type"] != "source":
            continue
        #
        X_train = dataset_data[source]["X"]
        y_train = dataset_data[source]["y"]
        #
        X_train, train_min, train_dim = normalize_train(X_train)
        #
        for target in dataset_data:
            if target == source:
                continue
            #
            if dataset_data[target]["type"] != "target":
                continue
            #
            print("Mode:",mode," Source:",source," Target:",target," Share:", dataset_data[target]["share"])
            #
            performance_data = []
            #
            for i in range(REPETITION_COUNT):
                if (i+1)%2 ==0:
                    print("Repetition: ", (i+1))
                #============================================================
                try:
                    X_target_train, y_target_train = dataset_data[target]["train"][i]
                    X_test, y_test = dataset_data[target]["test"][i]
                    #
                    X_target_train = normalize_test(X_target_train, train_min, train_dim)
                    X_test = normalize_test(X_test, train_min, train_dim)
                    #============================================================
                    X_train_join = pd.concat([X_train, X_target_train], axis=0, join="inner").reset_index(drop=True)
                    y_train_join = pd.concat([y_train, y_target_train], axis=0, join="inner").reset_index(drop=True)
                    #============================================================
                    train_pool_domain = [1 if e<len(X_train) else 0 for e in range(len(X_train_join))]
                except:
                    print("Error while preparing data")
                    continue
                #============================================================
                #print("TCA+")
                try:
                    _tca = tca_plus.TCAPlus(
                        test_set_domain = 0,
                        train_pool_domain = train_pool_domain,
                        test_set_X = X_test, 
                        train_pool_X = X_train_join, 
                        train_pool_y = y_train_join, 
                        Base_Classifier = LogisticRegression
                    )            
                    confidence, y_p = _tca.train_filter_test() 
                    accuracy, precision, recall, f1_score = calculate_results(y_test, y_p)

                    #Store results
                    data = ['TCA+', accuracy, precision, recall, f1_score, source, target]
                    performance_data.append(data)
                except:
                    print("Error while running TCA+")
                    continue
                #============================================================
                try:
                    X_m1_fit, X_m2_fit, y_m1_fit, y_m2_fit = train_test_split(X_train_join, y_train_join, test_size=0.8)
                    #============================================================
                    #REPD
                    #print("REPD")
                    autoencoder = AutoEncoder([48,24],0.01,100,50)
                    classifer = REPD(autoencoder)
                    classifer.fit(X_m1_fit, y_m1_fit)
                    y_p = classifer.predict(X_test)
                    accuracy, precision, recall, f1_score = calculate_results(y_test, y_p)

                    #Store results
                    data = ['REPD', accuracy, precision, recall, f1_score, source, target]
                    performance_data.append(data)

                    #REPD_EX
                    #print("REPDX")
                    final_classifier_1 = LogisticRegression(class_weight=cls_w_c1)
                    #
                    final_classifier_2 =  LogisticRegression(class_weight=cls_w_c2)
                    #
                    classifer = REPD_EX(base_repd=classifer, 
                                        defective_classification_model=final_classifier_1,
                                        non_defective_classification_model=final_classifier_2,
                                        use_def_m=use_c1,
                                        use_non_def_m=use_c2)
                    classifer.fit(X_m2_fit, y_m2_fit, train_base=False)
                    #
                    y_pb, y_p = classifer.predict(X_test)
                    accuracy_base, precision_base, recall_base, f1_base_score = calculate_results(y_test, y_pb)
                    accuracy, precision, recall, f1_score = calculate_results(y_test, y_p)

                    #Store results
                    data = ['REPD_EX', accuracy, precision, recall, f1_score, source, target]
                    performance_data.append(data)

                    #Close
                    autoencoder.close()                    
                except:
                    try:
                        autoencoder.close()
                    except:
                        pass
                    print("Error while running REPD and REPDX")
                    continue
            results_df = pd.DataFrame(performance_data, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 score', 'Source', 'Target'])
            results_df.to_csv("results/cross_"+mode+"_"+source+"_"+target)
            #
            print("Median:", results_df.groupby(["Model"])["F1 score", 'Precision', 'Recall'].median())    
            print()
            print()


from sklearn.model_selection import cross_val_score, KFold, train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

class RegressionModelsEvaluator():
    
    def __init__(self, partitions_on_data = 10, test_size= 0.20, color_palette=['red']):
        
        self.partitions_on_data = partitions_on_data
        self.test_size = test_size
        self.color_palette = color_palette

    def __prepare_data(self,data, output_var):
        
        partitions = self.partitions_on_data
        X  = data.drop(output_var,axis = 1).values
        y = data[output_var].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size, random_state = 7)
        kfold = KFold(n_splits = partitions, shuffle = True, random_state = 7 )
        
        return kfold, X_train, X_test, y_train, y_test
    
    
    def __generate_random_hex_colors(self, length):
        hex_colors = []
        for _ in range(length):
            hex_color = '#{:06x}'.format(random.randint(0, 0xFFFFFF))
            hex_colors.append(hex_color)
        return hex_colors
    
    def evaluate_models(self, models, data, output_var):
        names = []
        results = []
        
        
        np.random.seed(101)
        
        report_df = pd.DataFrame({}, columns= ['NAME','MSE', 'STD','RMSE'])
        kfold, X_train, X_test, y_train, y_test = self.__prepare_data(data, output_var)
        for name, model in models:
            cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
            results.append(np.sqrt(np.abs(cv_results)))
            names.append(name)

            #print results
            mse = abs(cv_results)
            u_mse = np.mean(mse)
            rmse = np.sqrt(mse)
            u_rmse = np.mean(rmse)
            sigma_rmse = np.std(rmse)
            
            msg = '%s\t\t\t   MSE %0.2f\t | STD:\t %0.2f\t | RMSE: \t %0.2f\t |' %(name, u_mse, sigma_rmse, u_rmse)
            print(msg)
            
            new_row = {
                'NAME': name, 
                'MSE': u_mse, 
                'STD': sigma_rmse, 
                'RMSE':u_rmse
            }
            report_df = report_df.append(new_row, ignore_index=True)
        
        #sort by the best to the worst model performance
        
        report_df['color'] = self.color_palette    
        report_df = report_df.sort_values(ascending=True, by='MSE')
        report_df.set_index('NAME', inplace=True)
        
        
        return names, results, report_df
    
    def generate_performance_report(self, names, results, report_df):
        
        #box plot setup
        
        fig = plt.figure(figsize = (15,10))
        fig.suptitle('Models Comparison by MSE/RMSE')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        plt.xticks(rotation=45)
        ax.set_xticklabels(names)
        plt.show()
        
        #Bar plot setup
        
        fig,axs = plt.subplots(1,2, figsize=(15,10))
        labels  = report_df.index.values
        
        # Error report
        tmp_df = report_df.sort_values(by='RMSE', ascending=False)
        colors = tmp_df['color'].values
        labels  = tmp_df.index.values
        tmp_df['RMSE'].plot(kind='bar',ax=axs[0],color= colors)
        axs[0].set_xticklabels(labels, fontsize=12)
        axs[0].set_xlabel('Model name', fontsize=15)
        axs[0].set_ylabel('Mean RMSE', fontsize=15)
        axs[0].set_title('Models performance measured by RMSE')
        
        #noise report
        
        tmp_df = report_df.sort_values(by='STD', ascending=False)
        colors = tmp_df['color'].values
        labels  = tmp_df.index.values
        tmp_df['STD'].plot(kind='bar',ax=axs[1],color= colors)
        axs[1].set_xticklabels(labels)
        axs[1].set_xlabel('Model name', fontsize=15)
        axs[1].set_ylabel('Mean Standard Error', fontsize=15)
        axs[1].set_title('Models performance measured by Noise')
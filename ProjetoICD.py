import pandas as pd
from scipy.io import arff
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import xlsxwriter

#code control panel
loadarff = False #load arff file (phased out)
loadcsv = True #load CSV file
plot = True #plot, only works with the csv file
gensum = False #generate summary stats
gensep = False #generate report on basic info on the data including kind and possible values
gencov = False #generate correlation matrix based on numpy covariance matrix

#plotting subset control panel
save_subsets_to_excel = False #saves grouped subsets of the dataframe as excel spreadsheets
generate_selected_graphs = False #generate choice graphs for visual KDD
Plot_categorical = False #plot all categorial data with respect to class
Plot_numerical = True #plot all numerical data with respect to class
printclassification = False # determines whether to print or not class by item grouped by dataframes

non_numerical_list = [] #currently not being used

#functions
def normalize(df_csv, csv_cols ):
    '''
    Normalize the dataframe
    :param df_csv: dataframe to be normalized
    :param csv_cols: column names passed as a list
    :return: a normalized data frame
    '''
    for col in csv_cols:
        maxx = df_csv[col].max()
        try:
            df_csv[col] /= maxx
        except:
            # this here is in case normalized is applied to a Nan arg
            print("Normalize function failed, argument of type {} passed" , type(df_csv[col])) 
    return df_csv
def generate_mat(df,csv_cols):
    '''
    Generates a matrix off of applying a function on all possible parameter combinations. 
    Also, just because it can compute it, does not mean that all parameters make sense since
    numerical categoric data will "succeed" despite making no sense. 
    :param df: the data frame to be analyzed
    :param csv_cols: df's column list
    :return: a dataframe with all possible combinations with the given function
    '''
    df_out = pd.DataFrame(index=csv_cols, columns=csv_cols)
    df_out = df_out.fillna(0)
    for col in csv_cols:
        for col1 in csv_cols:
            if col1 == col:
                df_out.loc[col, col1] = 0
            else:
                try:
                    df_out.loc[col, col1] = np.cov(df[col], df[col1])[0,1]
                except:
                    df_out.loc[col, col1] = 0
    return df_out
def save_to_txt(base_filename, working_folder, data_frame):
    '''
    saves a dataframe to a txt file
    :param base_filename: name of the file you want to create
    :param working_folder: the folder where you want to store it
    :param data_frame: the dataframe you want to save
    :return: nothing, it just saves the data frame
    '''
    with open(os.path.join(working_folder, base_filename), 'w') as outfile:
        data_frame.to_string(outfile)
    return
def cleanup(df):
    '''
    Cleans up any matrices stored as dataframes from as much 0s as possible
    :param df: dataframe to be cleaned
    :return: a the input dataframe post cleaning
    '''
    # cleanup of 0's and null
    df = df[(df != 0).any()]
    df = df[(df.T != 0).any()]
    for col in df.columns:
        if df[col].sum() == 0:
            del df[col]
    return df

if loadarff:
    #load ARFF file
    fileName = "credit.arff"
    fileLocation = "C:\AAAUserFiles\\"
    data = arff.loadarff(fileLocation + fileName)
    df_arff = pd.DataFrame(data[0])
    cols = df_arff.columns

if loadcsv:
    #Load file as CSV (working)
    fileName = "credit.csv"
    fileLocation = "C:\AAAUserFiles\\"
    df_csv = pd.read_csv(fileLocation+fileName, sep = ";")
    printPreview = False
    if printPreview:
        print(df_csv.head())
        print(df_csv.shape)
        print(df_csv.columns)
        print(df_csv.dtypes)

if gencov:
    # covariance attempt. Generate pearsons_r from covariance matrix
    csv_cols = df_csv.columns
    df_cov = pd.DataFrame(index=csv_cols, columns=csv_cols)
    df_csv1 = pd.DataFrame().reindex_like(df_csv)
    df_cov = df_cov.fillna(0)
    
    df_csv1 = normalize(df_csv, csv_cols)
    df_cov = generate_mat(df_csv1,csv_cols)
    df_cov = cleanup(df_cov)
    save_to_txt('Covariance.txt', 'C:\AAAUserFiles', df_cov)

if gensep:
    # creates list of numeric attributes and non numeric
    numeric = []
    nonnumeric = {}
    #separating numeric from categorical/non numeric
    unique = pd.DataFrame()
    df_inp = pd.DataFrame()
    if loadcsv:
        df_inp = df_csv
        cols = df_csv.columns
    elif loadarff:
        df_inp = df_arff
        cols = df_arff.columns
    else:
        print("error in gensep")

    for col in cols:
        ev = pd.Series(df_inp[col].unique())
        if len(ev) > 10:
            numeric.append(col)
        else:
            nonnumeric[col] = ev
            #print("Name of Attribute: " + col + " , unique values:\n", ev)
    #output a value summary
    lista = []
    for key,value in nonnumeric.items():
        lista.append(key)
    non_numerical_list = lista
    
    if False:
        print("List of numeric values: {}", format(numeric))
        print("List of non-numeric arguments: {}", format(lista))
    
    #save_to_txt('numéricos_vs_categóricos.txt', 'C:\AAAUserFiles', unique)

if plot & loadcsv:
    if generate_selected_graphs:
        # list of cols of interest, personal_status vs num_dependents, own_telephone vs. num_dependents, housing
        # own_telephone vs. other_payment_p, num_dependents vs. personal_status, credit_ammount vs. Age,
        # credit_ammount vs. job, credit_ammount vs. duration.
        sns.set_style("darkgrid")
        sns.set_palette("dark")
        sns.despine(left=True, bottom=True)
        plot_vector = [("personal_status", "num_dependents"), ("own_telephone", "num_dependents"), ("housing", "housing"), \
                       ("own_telephone", "other_payment_plans")]
        folder = "C:\AAAUserFiles\\1 Data Science and Engineering\Projeto Final ICD\Graficos\\"
        for a, b in plot_vector:
            myPlot = sns.stripplot(x=df_csv[a], y=df_csv[b], hue=df_csv["class"], palette="rocket")
            plt.savefig(folder + a + " x " + b + '.png')
            plt.show()

        # item list
        
    #dataframe column list for categorical plotting and numerical plotting, separated
    criterion = ['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment' \
        , 'installment_commitment', 'personal_status', 'other_parties', 'residence_since', \
                 'property_magnitude', 'other_payment_plans', 'housing', 'existing_credits', 'job', \
                 'num_dependents', 'own_telephone', 'foreign_worker']
    number = {'duration' : "Duration of Loan in Months" ,\
              'credit_amount' : "Amount of Credit in German Marks",\
              'age' : "Age of Loan Recipient"}
        
    #Setting Seaborn style and context
    sns.set_context("paper")
    sns.set_theme()
    sns.set_style()
    sns.color_palette("rocket")
    folder = "C:\AAAUserFiles\\1 Data Science and Engineering\Projeto Final ICD\Graficos\\"
    
    #fixing the categorical number encoded entities for graphing
    df_csv["num_dependents"] = df_csv["num_dependents"].replace([1, 2], ["one", "two"])
    df_csv["residence_since"] = df_csv["residence_since"]\
        .replace([1, 2, 3, 4],["one year or less", "two years", "three years","four or more years"])
    df_csv['installment_commitment'] = df_csv['installment_commitment']\
        .replace([1, 2, 3, 4], ["group one", "group two","group three","group four"])
    df_csv['existing_credits'] = df_csv['existing_credits']\
        .replace([1, 2, 3, 4], ["one", "two", "three", "four"])
    
    #plot all categorical data
    if Plot_categorical:
        for var in criterion:
            print(var)
            plt.figure()
            p = sns.histplot(data=df_csv, x=var, hue="class",
                             multiple="dodge", shrink=0.7, palette="rocket" ,
                                 bins = len(df_csv[var].unique()))
            plt.xticks(rotation=45, fontsize = 12 , fontweight = 'semibold' , horizontalalignment = "right")
            plt.yticks(fontsize = 12 , fontweight = "semibold")
            plt.xlabel("")
            plt.ylabel("count" , fontsize = 16 , fontweight = "semibold")
            sns.despine(bottom=True )
            plot_name = var.replace("_", " ").title()
            plt.title(plot_name, {'fontsize': 30, 'fontweight': "bold"})
            plt.tight_layout(h_pad=2)
            plt.savefig(folder + var + '.png')
            #plt.show()

    #plot all numerical data
    sns.set_context("paper")
    for num,name in number.items():
        fig, (ax1, ax2) = plt.subplots(nrows = 1,ncols = 2, sharex = True , figsize = (12,8))
        plot_name = num.replace("_", " ").title()
        fig.suptitle("   " + plot_name, fontsize = 30, fontweight = "bold" , ha = "center")
        fig.supxlabel("   " + name, fontsize=18, fontweight="bold", ha = "center" , va = "bottom")
        sns.histplot(data=df_csv, x=num, hue="class", multiple="stack" , ax=ax1 )
        sns.boxplot(data=df_csv, x=num, hue="class", ax=ax2)
        
        plt.setp(ax1.get_legend().get_texts(), fontsize='16' )
        plt.setp(ax1.get_legend().get_title(), fontsize='20')
        ax1.margins(tight = True)
        
        if max(df_csv[num]) > 1000 :
            mag = np.floor(np.log10(max(df_csv[num]))-0.6)
            randvar = np.ceil(max(df_csv[num]) + 1)
            tick_vector = np.arange(0,((randvar - randvar%10**np.floor(np.log10(randvar)))*2).astype(int),\
                                    ((np.floor(10 ** (np.log10(max(df_csv[num])) - 1) / 10 ** mag)* 10 ** mag)*2).astype(int))

        else:
            tick_vector = np.arange(0,max(df_csv[num])+1,10)
            
        ax1.set_xticks(tick_vector)
        ax1.set_xticklabels(tick_vector,fontsize = 18, fontweight = "bold" , rotation=45)
        ax2.set_xticks(tick_vector)
        ax2.set_xticklabels(tick_vector, fontsize=18, fontweight="bold" , rotation=45)
        ax1.set_xlabel("")
        ax2.set_xlabel("")

        if max(df_csv[num].value_counts()) < 100:
            tick_vector1 = np.arange(0,200,20)
        else:
            tick_vector1 =np.arange(0, max(df_csv[num].value_counts()) + 1, 10)

        ax1.set_yticks(tick_vector1)
        ax1.set_yticklabels(tick_vector1, fontsize=18, fontweight="bold")
        ax1.set_ylabel("Count" , fontsize = 18, fontweight = "bold" )
        sns.despine(bottom=True)
        plt.tight_layout(h_pad=0)
        plt.savefig(folder + "Numeric Plot! " + num + '.png')
        print(num)
        plt.show()
        
    print( "graphs generated successfully")
   
    #save grouped dataframe data to excel
    if save_subsets_to_excel:
        writer = pd.ExcelWriter('C:\AAAUserFiles\output.xlsx', engine='xlsxwriter')
        count = 0
        for item in criterion:
            count +=1
            df_new = pd.DataFrame(data = df_csv.groupby(item)["class"].value_counts()).unstack()
            df_new.columns = ['_'.join(col) for col in df_new.columns]
            df_new = df_new.rename(columns={"class_bad": "bad", "class_good": "good" })
            df_new = df_new.T
            print(df_new, "\n\n")
            sheet_name = "Sheet" + str(count)
            df_new.to_excel(writer, sheet_name=sheet_name)
            
            # save the excel
            writer.save()
            print('DataFrame is written successfully to Excel File.')
   
if gensum:
    # creates attribute summaries and saves them to a file
    #creating the descriptions
    if (loadarff & ~loadcsv):
        summ = df_arff.describe()
    elif(~loadarff & loadcsv):
        summ = df_csv.describe()
    elif(loadarff & loadcsv):
        summ = df_arff.describe()
    else:
        print("An error occured, neither file has been opened")
    #creating the save file
    save_to_txt('atributos.txt', 'C:\AAAUserFiles', summ)

print("code ran successfully")

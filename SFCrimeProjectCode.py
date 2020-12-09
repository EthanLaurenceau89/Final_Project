import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


CrimeData = pd.read_csv("SFCD_2018.csv")

def CrimeFunc(data):                 ## Function that will take in crime dataset. 
  return data.head()                 ## Use head command to retrieve first five rows of data, to get a sense of what the data looks like.


Crime_Count = CrimeData['Incident Description'].value_counts().head(20)  #Retrieve the top 20 incindents from SF Crime dataset

def top20Incidents(data):                                   # Function Constructs a DataFrame/table from top 20 crimes in San Francisco and                                                               the number of times they have occured.
   Crime_df = pd.DataFrame(Crime_Count).reset_index()      
   Crime_df.columns = ["Crime", "Occurences"]
   return Crime_df


def plottop20(crimecount):                                   # Creates a bar plot, that shows each of top 20 crimes, vs. # of occurences.
    sns.set_style("whitegrid")
    plt.title("Occurences of Crimes in San Francisco")
    crimecount.plot(kind='bar')
    plt.show()


def plot_week_occurence(data):                               # Function will plot the total occurence of crimes from dataset by days of                                                                    week.
    
    
    CrimePerDay = pd.concat([data["Incident Day of Week"], data["Incident Description"]], axis = 1)   # Create a table that lists Weekdays                                                                                                           with the number of crime occurences.                                                                                                         This is done by concatenating                                                                                                               "Incident Day of the Week with the                                                                                                           Incident Description".
    
    CrimeDayCt = CrimePerDay["Incident Day of Week"].value_counts().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])                                                                              # What I do hear, is count how many                                                                                                         times that the Weekday shows up in                                                                                                           the Dataset, b/c this gives me how                                                                                                           many crimes occured for each weekday.
    print(CrimeDayCt)


    plt.figure(figsize = (7,7))                                           # Plot the figure for the number of crimes that occur per weekday.
    plt.title("Occurences of Crimes in San Francisco", fontstyle = "oblique", fontsize = 14)
    CrimeDayCt.plot(kind = "bar", color = ['r','g','b','y', 'orange','purple', 'black'], fontsize = 13) #Set colors of bars for each weekday
    plt.show()

    
def IncidentbyMonth(data):                                                  
                                                                         # Retrieve the Month, Day, Hour for each of the incidents, and                                                                                 place construct respective column for them in the dataset.
    
    data["Incident Date"] = pd.to_datetime(data["Incident Date"])            
    data['Incident Month'] = pd.DatetimeIndex(data['Incident Date']).month
    data["Incident Date"] = pd.to_datetime(data["Incident Date"])
    data['Incident Day'] = pd.DatetimeIndex(data['Incident Date']).day
    data['Incident Hour'] = pd.DatetimeIndex(data['Incident Time']).hour
                                                                          # Look at number of crimes for each month,, and then group them by                                                                             the incident years which are 2018 and 2019.
    
    Month_data = pd.DataFrame(data.groupby(["Incident Month"])["Incident Year"].value_counts())
    Month_data.index = pd.MultiIndex.from_tuples([('Jan.', 2018), ('Jan.', 2019), ('Feb.', 2018), ('Feb.', 2019), ('Mar.', 2018), ('Mar.', 2019), ('Apr.', 2018), ('Apr.', 2019), ('May', 2018), ('May', 2019), ('Jun.', 2018), ('Jun.', 2019), ('Jul.', 2018), ('Jul.', 2019), ('Aug.', 2018), ('Sep.', 2018), ('Oct.', 2018), ('Nov.', 2018), ('Dec.', 2018)])
    Month_data.columns = ["Incident Occurences"]
    return Month_data

    
def plotIncidentbyMonth(data):                                    #The plots the number of crimes for each month reported for 2018 and 2019

    plt.title("Incident Occurenes in SF From 2018 - Jul. 1, 2019")
    M = sns.countplot(data=data, x = "Incident Month", hue = "Incident Year")
    M.set_xticklabels(['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.'])
    M.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    
def GetandPlotResolution(data):                         # This function will plot a pie chart from the Resolution for the Incidents                                                                    documented in the Dataset.                                                        
    
    Res_Vals = data["Resolution"].value_counts()        # First it takes in the counts of the Resolution
    print(Res_Vals)
                                                        # The below code is used to retrieve the districbution of the resolutions for the                                                             incidents in the dataset.
        
                                                        # These percentages will be used for the legend for my pie chart, they will be                                                                 placed  with names of the Resolutions to be plotted.
    
    CrimeRes = pd.DataFrame()                           
    CrimeRes["Resolution"] = list(Res_Vals.index)
    CrimeRes["Occurence"] = list(Res_Vals)
    CrimeRes["Percentages"] = (CrimeRes["Occurence"] / CrimeRes["Occurence"].sum() * 100).round(2)
    CrimeRes["Percentages2"] = list(CrimeRes["Percentages"].astype(str))
    CrimeRes["Res"] = CrimeRes["Resolution"] + " (" + CrimeRes["Percentages2"] + "%)"

                                                          # This code actually plots the piechart for the Resolutions of Crimes in San                                                                   Franciso documented in the dataset with a legend.
    
    plt.figure(figsize =(11,11))                       
    plt.pie(list(CrimeRes["Percentages"]))
    plt.title("Crime Resolutions in San Franciso", fontsize = 14)
    plt.axes().set_ylabel('')
    plt.legend(CrimeRes["Res"], loc='best')
    plt.show()
    
                                                          # Now I'm in the stage of training my data and fitting a SVC model so that based                                                              on certain parameter which I will specify below, I can predict how the crime will                                                            be resolved.
        
                                                          # I'll import the modules from sklearn I'll need to train, fit, and retrieve the                                                               confusion matrix and classification report from applying SVM to crime data from                                                             certain parameters.
    
from sklearn.model_selection import train_test_split                              
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import SVC

    
def SupVecMach(data):                                     # Function used to train data and apply the SVM Model to predict Resolution, based                                                           on  The Weekday, The Month, The Day number, The hour, and Longitude and Latitude                                                             coordinates for which the crime occured.
    
    
    CrimeData2 = data[["Incident Day of Week", "Latitude", "Longitude", "Incident Month", "Incident Hour", "Incident Day", "Resolution"]]
    CrimeData2.dropna(inplace = True)                                     # Remove Nan values so that I can use SVM model
                                                                          # Set dummy variables to Weekday Values, and Resolution Names                                                                                 because SVM model only takes in  numeric types.
    CrimeData2.replace(list(CrimeData2["Incident Day of Week"].value_counts().index), range(7), inplace = True)   
    CrimeData2.replace(list(CrimeData2["Resolution"].value_counts().index), range(6), inplace = True)
    X = CrimeData2
    y = X['Resolution']
    X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

    
    svc_model = SVC()
    svc_model.fit(X_train,y_train)

    predictions = svc_model.predict(X_test)
                                                          # Here I output the Confusion Matrix and Classification Report for SVM model.

    print("Confusion Matrix\n", confusion_matrix(y_test,predictions), "\n", sep = "\n")
    print("Classification Report for Resolution\n", classification_report(y_test,predictions), "\n", sep ="\n")
    
    return X
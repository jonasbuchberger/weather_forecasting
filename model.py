import netCDF4 as nc
from netCDF4 import Dataset
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

#Possible Points with given dataset
# lat : [ 31.  32.  33.  34.  35.  36.  37.  38.  39.]
# lon : [ 254.  255.  256.  257.  258.  259.  260.  261.  262.  263.  264.  265. 266.  267.  268.  269.]

#print alphas and coefs
def print_alpha(ridge, x, y, alphas):
    coefs = []
    for a in alphas:
        ridge.set_params(alpha=a)
        ridge.fit(x, y)
        coefs.append(ridge.coef_)

    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')
    plt.show()

#Loads NetCDF4 files given in files and sub_strings for a given lat and lon
def loadNetCDF4(files, sub_str, size, lat, in_lon):

    matrix = np.empty((size,))

    for i, item in enumerate(files):
        print 'Loading and avg netCDF4 weather data' + files[i]
        data = Dataset(files[i] + sub_str)
        
        precip = data.variables.values()[-1]

        lat = data.variables['lat']
        l = lat[:]

        array_lat = np.where(l == lat)[0][0]

        lon = data.variables['lon']
        n = lon[:]

        array_lon = np.where(n == in_lon)[0][0]

        #Get the data to a given point
        p_lat_lon = precip[:,:,:,array_lat,array_lon]

        #Build average values of 11 ridges over time 
        p = np.mean( p_lat_lon, axis = 1 )

        #Build average of 3 time steps
        p = np.mean(p, axis = 1)

        #Merge with other variables
        matrix = np.column_stack((matrix, p))
    
        data.close()

    return matrix



def main(lat, lon, station_index): 

    files = ['dswrf_sfc','dlwrf_sfc','uswrf_sfc','ulwrf_sfc','ulwrf_tatm','pwat_eatm','tcdc_eatm','apcp_sfc','pres_msl','spfh_2m','tcolc_eatm','tmax_2m','tmin_2m','tmp_2m','tmp_sfc']
    train_sub_strings = '_latlon_subset_19940101_20071231.nc'
    #test_sub_str = '_latlon_subset_20080101_20121130.nc'  

    #Load csv Solar Energy
    print 'Importing solarenergy trainings data'
    energy = np.genfromtxt('train.csv', delimiter=',', dtype="float")
    energy = np.squeeze(energy[:,station_index])
    energy = np.delete(energy, 0, 0)

    #Split in train and test data
    print 'Splitting solarenergy data in test and train data'
    energy_split = np.split(energy,[4018,5113])
    train_energy = energy_split[0]
    test_energy = energy_split[1]

    #Loading netCDF4 data for a specific point(lat,lon)
    train_matrix = loadNetCDF4(files, train_sub_strings, 5113, lat, lon)

    #Deleting zero colum 
    train_matrix = np.delete(train_matrix, 0, 1)

    #Split in train and test data
    print 'Splitting weather data in test and train data'
    train_split = np.split(train_matrix,[4018,5113])
    train_matrix = train_split[0]
    test_matrix = train_split[1]

    #Build csv train
    np.savetxt(str(lat) + '_' + str(lon) + '_' + str(station_index) + '_train.csv', train_matrix, delimiter = ",", fmt = "%.06f" )

    print 'Setting up Regressor'
    ridge = Ridge()

    #Prepare a range of alpha values to test
    alphas = np.array([1,0.1,0.01,0.001,0.0001,0])

    #Printing alphas, taken form scikit
    #print_alpha(ridge, train_matrix, train_energy, alphas)

    # create and fit a ridge regression model, testing each alpha, taken from scikit
    grid = GridSearchCV(estimator=ridge, param_grid=dict(alpha=alphas))
    grid.fit(train_matrix, train_energy)
    print 'Best estimated alpha: '
    print(grid.best_estimator_.alpha)
    ridge.alpha=grid.best_estimator_.alpha

    print 'Training the Regressor'
    ridge.fit(train_matrix,train_energy)

    print 'Predicting Energy'
    prediction_matrix = ridge.predict(test_matrix)

    #Sace csv prediction
    np.savetxt( str(lat) + '_' + str(lon) + '_' +str(station_index) + '_prediction.csv', prediction_matrix, delimiter = ",", fmt = "%d" )

    #Plotting

    #Setting up date x-axis
    time = pd.date_range('2005-01-01', periods=1095)#1095

    fig, ax = plt.subplots(1)
    fig.autofmt_xdate()
    xfmt = mdates.DateFormatter('%d-%m-%y')
    ax.xaxis.set_major_formatter(xfmt)

    #Plot prediction and actual values
    ax = plt.gca()
    ax.plot(time, prediction_matrix, linewidth=0.5)
    ax.plot(time, test_energy, linewidth=0.5,)

    #Labels and Legend
    plt.xlabel('Time')
    plt.ylabel('Jouls per square meter')
    plt.title('Solar Energy of Tahlequah(Oklahoma)' + ' (lat: ' + str(lat) + ' lon: ' + str(lon-360) + ')' )

    plt.axis('tight')

    prediction_patch = mpatches.Patch(color='blue', label='Prediction')
    meassured_patch = mpatches.Patch(color='orange', label='Meassured')
    plt.legend(handles=[prediction_patch, meassured_patch])

    plt.show()

    #Plot difference graph
    differnece = np.subtract(prediction_matrix, test_energy)

    fig, ax = plt.subplots(1)
    fig.autofmt_xdate()
    xfmt = mdates.DateFormatter('%d-%m-%y')
    ax.xaxis.set_major_formatter(xfmt)

    ax = plt.gca()
    ax.plot(time,differnece, linewidth = 0.5)
    #ax.plot(time, np.full((1095,),4000000), color='orange')
    #ax.plot(time, np.full((1095,),-4000000), color='orange')
    plt.xlabel('Time')
    plt.ylabel('Jouls per square meter')
    plt.title('Solar Energy of Tahlequah(Oklahoma)' + ' (lat: ' + str(lat) + ' lon: ' + str(lon-360) + ')' )
    plt.axis('tight')
   
    difference_patch = mpatches.Patch(color='blue', label='Difference')
    plt.legend(handles=[difference_patch])

    plt.show()


#Go into the station.csv and get the station_index for your lat and lon
# main(lat, lon, station_index)
main(36, 360-95, 86)

#Get all necessary data form -- https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest/data
#Little bit of help and inspiration from -- http://fastml.com/predicting-solar-energy-from-weather-forecasts-plus-a-netcdf4-tutorial/   
#Extract all in the directory of the script
#Run the script
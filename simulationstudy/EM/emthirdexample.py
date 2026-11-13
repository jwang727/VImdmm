import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from collections import Counter

import math
import csv

from scipy.stats import multivariate_normal, invwishart, wishart, dirichlet, rv_discrete, beta, gamma

import sys
print(sys.version)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataseedtrain", type=int, required=True)
parser.add_argument("--N", type=int, required=True)
parser.add_argument("--q", type=int, required=True) 
parser.add_argument("--Ktrue", type=int, required=True) 
args = parser.parse_args()

dataseedtrain = args.dataseedtrain
N = args.N
q = args.q 
Ktrue = args.Ktrue 
print(f"Running with dataseedtrain = {dataseedtrain} and N = {N} and q ={q} and Ktrue={Ktrue}") 

filenamesuffix=str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue) 

p=1 
d_1=q 

maxdist=Ktrue*4 
maxvar=(maxdist*0.4)**2 

param_rng = np.random.default_rng(seed=13579) 

def checkdist(new_point, points, r_threshold):
    for point in points:
        dist = np.sqrt(np.sum(np.square(new_point-point)))
        if dist < r_threshold:
            return False


    return True


def Randvec(Nvec, L, r_threshold):
    points = []
    scope = np.arange(0,L,0.1)
    while len(points) < Nvec:
        new_point = param_rng.choice(scope, q)
        print('old_points')
        print(points)
        print('new_point')
        print(new_point)
        if checkdist(new_point, points, r_threshold):
            points.append(new_point)

    return points


def Randmat(maxvar):

    sampleunif=param_rng.uniform(low = maxvar/2, high = maxvar, size = q)
    sampleunif=np.round(sampleunif, decimals=1)

    from scipy.stats import ortho_group  

    U = ortho_group.rvs(dim=q, random_state=param_rng)

    return U @ np.diag(sampleunif) @ np.transpose(U)


def multipleRandmat(Nmat,maxvar):
    matriceslist=[]

    while len(matriceslist) < Nmat:
        new_mat = Randmat(maxvar)
        matriceslist.append(new_mat)
    return matriceslist


truemu_kg=np.zeros([Ktrue,1,q])

for k in range(0,Ktrue):

        truemu_kg[k,0,k]=maxdist

        print('k is '+str(k))


trueSigma_k=np.zeros([Ktrue,1,q,q])

for k in range(0,Ktrue):

        trueSigma_k[k,0,:]=multipleRandmat(1,maxvar)[0]


trueclusterprob = param_rng.uniform(1 / (2 * Ktrue), 2 / Ktrue, Ktrue)
trueclusterprob /= trueclusterprob.sum()

categorysizes=[d_1] 
truepsi=np.zeros([Ktrue,p]+[d_1]) 

highprob=0.75
truepsi[:,0,:]=(1-highprob)/(d_1-1)+np.zeros((Ktrue,d_1)) 
for k in range(0,Ktrue):
    truepsi[k,0,k] = highprob


def generate_data_contcase(N,Ktrue,truepsi,truemu_kg,trueSigma_k,trueclusterprob, data_seed):

    data_rng = np.random.default_rng(seed=data_seed)

    component = data_rng.choice(Ktrue, size=N, p=trueclusterprob)

    categoricaldata = np.zeros((N,p), dtype=int)

    for i in range(N):
        for j in range(p):

            print('check sampled items')
            print(np.arange(categorysizes[j]))
            print('check prob vec')
            print(truepsi[component[i],j,:])
            categoricaldata[i, j] = data_rng.choice(np.arange(categorysizes[j]), p=truepsi[component[i],j,:] )


    continuousdata=np.zeros([N,q])

    for i in range(0,N):

        x = data_rng.multivariate_normal(truemu_kg[component[i],0,:], trueSigma_k[component[i],0,:], size=1)

        continuousdata[i,:]=x


    return continuousdata, categoricaldata, component


continuousdata, categoricaldata, component=generate_data_contcase(N,Ktrue,truepsi,truemu_kg,trueSigma_k,trueclusterprob, data_seed=dataseedtrain)

fulldatamat=np.hstack((continuousdata,np.reshape(categoricaldata,(N,p)), np.reshape(component,(N,1)) )) 

contcol_list = [f'contcol_{i}' for i in range(1, q+1)]
catcol_list=[f'catcol_{i}' for i in range(1,p+1)]

fulldata=pd.DataFrame(fulldatamat, columns = contcol_list+catcol_list+['clusterlabel'])   

discrete_cols=catcol_list+['clusterlabel'] 

for col in discrete_cols:
    fulldata[col] = fulldata[col].astype(int)


allcategoricalvalues=[0,1]

modelled_continuous_columns=contcol_list 
modelled_categorical_columns=catcol_list

def merge_columns(row):
    return '_'.join(row.astype(str))


fulldata['combinedcategories']=fulldata[modelled_categorical_columns].apply(merge_columns, axis=1)

modelled_categorical_columns_combined=['combinedcategories']

fulldataframe=pd.DataFrame(fulldata, columns = modelled_continuous_columns+modelled_categorical_columns+modelled_categorical_columns_combined+['clusterlabel'])

scaler = StandardScaler()

fulldataframe[modelled_continuous_columns] = scaler.fit_transform(fulldataframe[modelled_continuous_columns])

scaler.fit_transform(np.array(fulldata[modelled_continuous_columns])) 

combinedcategories=np.array(fulldataframe['combinedcategories'])

continuousdata=np.array(fulldataframe[modelled_continuous_columns])

allcatvals=np.unique( np.array([str(i) for i in np.arange(d_1)], dtype='object') , return_counts=True) 

fulldataframegroupbymeans = fulldataframe.groupby(['combinedcategories'], as_index=False)[modelled_continuous_columns].mean() 

fulldataframegroupbystd = fulldataframe.groupby(['combinedcategories'], as_index=False)[modelled_continuous_columns].std() 

categorical_columns_dims=[]

for catcols in modelled_categorical_columns:

    categorical_columns_dims.append(d_1) 


print('done simulating data')

print('check true parameter values')

print('truemu_kg')
print(truemu_kg)

print('trueSigma_kg')
print(trueSigma_k)

print('truepsi')
print(truepsi)

print('trueclusterprob')
print(trueclusterprob)

Ntest=min(int(N*0.4),2000)

testcontinuousdata, testcategoricaldata, testcomponent=generate_data_contcase(Ntest,Ktrue,truepsi,truemu_kg,trueSigma_k,trueclusterprob, data_seed=dataseedtrain+1000)

testdatamat=np.hstack((testcontinuousdata,np.reshape(testcategoricaldata,(Ntest,p)), np.reshape(testcomponent,(Ntest,1)) ))

contcol_list = [f'contcol_{i}' for i in range(1, q+1)]

testdata=pd.DataFrame(testdatamat, columns = contcol_list+catcol_list+['clusterlabel'])   

for col in catcol_list+['clusterlabel']:
    testdata[col] = testdata[col].astype(int)


testdata['combinedcategories'] = testdata[modelled_categorical_columns].apply(merge_columns, axis=1)

testdataframe=pd.DataFrame(testdata, columns = modelled_continuous_columns+modelled_categorical_columns+modelled_categorical_columns_combined+['clusterlabel'])

testdataframe[modelled_continuous_columns] = scaler.transform(testdataframe[modelled_continuous_columns].to_numpy()) 

testcontinuousdata=np.array(testdataframe[modelled_continuous_columns])

print('done')

import pickle

outputtruemu_kg = open('truemu_kg'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(truemu_kg, outputtruemu_kg)
outputtruemu_kg.close()

outputtrueSigma_k = open('trueSigma_k'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(trueSigma_k, outputtrueSigma_k)
outputtrueSigma_k.close()

outputtruepsi = open('truepsi'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(truepsi, outputtruepsi)
outputtruepsi.close()

outputtrueclusterprob = open('trueclusterprob'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(trueclusterprob, outputtrueclusterprob)
outputtrueclusterprob.close()

outputfulldata=open('trainingdata'+filenamesuffix+'.pkl','wb') 
pickle.dump(fulldata,outputfulldata)
outputfulldata.close()

outputtestdata=open('testdata'+filenamesuffix+'.pkl','wb') 
pickle.dump(testdata,outputtestdata)
outputtestdata.close()

outputscaler=open('scaler'+filenamesuffix+'.pkl','wb') 
pickle.dump(scaler,outputscaler) 
outputscaler.close() 

print('done saving true params')

Psiindependentofk=0

catvaluetoindex = dict(zip(allcatvals[0], list(range(0, len(allcatvals[0])))))

mudependentong=0
Sigmadependentong=0
pidependentong=0

if mudependentong==0:
    catvaluetoindexmu = dict(zip(allcatvals[0], [0] * len(allcatvals[0]) )) 

    allcatvalsmu=np.array(['0'])

    catvaluetoindexmu['0']=0
    combinedcategoriesmu=np.array(['0']*N)
else:
    allcatvalsmu=allcatvals
    catvaluetoindexmu = dict(zip(allcatvals[0], list(range(0, len(allcatvals[0])))))
    combinedcategoriesmu=combinedcategories

if Sigmadependentong==0:
    catvaluetoindexSigma = dict(zip(allcatvals[0], [0] * len(allcatvals[0]) ))
    allcatvalsSigma=np.array(['0'])

    catvaluetoindexSigma['0']=0
    combinedcategoriesSigma=np.array(['0']*N)
else:
    allcatvalsSigma=allcatvals
    catvaluetoindexSigma = dict(zip(allcatvals[0], list(range(0, len(allcatvals[0])))))
    combinedcategoriesSigma=combinedcategories

if pidependentong==0:
    catvaluetoindexpi = dict(zip(allcatvals[0], [0] * len(allcatvals[0]) ))
    allcatvalspi=np.array(['0'])

    catvaluetoindexpi['0']=0
    combinedcategoriespi=np.array(['0']*N)
else:
    allcatvalspi=allcatvals
    catvaluetoindexpi = dict(zip(allcatvals[0], list(range(0, len(allcatvals[0])))))
    combinedcategoriespi=combinedcategories

print('check mu catval variables')
print(allcatvalsmu)
print(catvaluetoindexmu)
print(Counter(combinedcategoriesmu))

print('check Sigma catval variables')
print(allcatvalsSigma)
print(catvaluetoindexSigma)
print(Counter(combinedcategoriesSigma))

print('check pi catval variables')
print(allcatvalspi)
print(catvaluetoindexpi)
print(Counter(combinedcategoriespi))

from scipy.stats import multivariate_normal, invwishart, wishart, dirichlet, rv_discrete, beta, gamma

K = Ktrue
p = len(modelled_categorical_columns)
q = len(modelled_continuous_columns) 

checkrow = 3
print(combinedcategories[checkrow])
print(catvaluetoindex[combinedcategories[checkrow]])

iterations = 3001 
burn_in = int(iterations * 0.2)

mdims = tuple([iterations,K,q])

zparamdims=tuple([N, K])

r_ik=np.zeros(zparamdims)+0.1/(0.9+0.1*(K-1))

rho_ik=np.zeros(zparamdims)
logrho_ik=np.zeros(zparamdims)

from kmodes.kprototypes import KPrototypes

cat_indices = [fulldataframe[modelled_continuous_columns+modelled_categorical_columns].columns.get_loc(col) for col in modelled_categorical_columns] 

kproto = KPrototypes(n_clusters=K, init='Cao', n_jobs=-1, random_state=42)
kprotoclusters = kproto.fit_predict(fulldataframe[modelled_continuous_columns+modelled_categorical_columns].to_numpy(), categorical=cat_indices)

for i in range(0,N):

    r_ik[i,kprotoclusters[i]]=0.9/(0.9+0.1*(K-1))


mu = np.zeros(mdims)  

Sigma = np.zeros((iterations, K, q, q))

piprobdim = tuple([iterations,K])

print('check piprobdim')
print(piprobdim)

piprob = np.zeros(piprobdim)  

Psidims = tuple([iterations, K, p] + [np.max(categorical_columns_dims)])

print('check Psidims')
print(Psidims)

Psi = np.zeros(Psidims)  

print('check piprob')
print(piprob[0, :])
print('check Psi')
print(Psi[0, :])
print('check mu')
print(mu[0, :])
print('check Sigma')
print(Sigma[0, :])

import pickle
import time

from scipy.special import logsumexp

start_time = time.time()

categorical_int_data = fulldataframe[modelled_categorical_columns].to_numpy().astype(int) 

def computelikelihood(mu,Sigma,Psi,pi):

    gauss_ll = np.zeros((N,K)) 
    for k in range(0,K): 
        gauss_ll[:,k] = multivariate_normal.logpdf(x=continuousdata, mean=mu[k,:], cov=Sigma[k,:]) 


    Psi_ll = np.zeros((N,K)) 
    for j in range(0,p): 

        Psi_ll = Psi_ll + np.log(Psi[:, j, categorical_int_data[:, j]]).T 


    llh = np.log(pi)[np.newaxis, :] + gauss_ll + Psi_ll 

    log_likelihood_per_sample = logsumexp(llh, axis=1)
    total_log_likelihood = np.sum(log_likelihood_per_sample)

    return total_log_likelihood, log_likelihood_per_sample, llh


timerecord=np.zeros(iterations)
em_compute_elapsed=0.0 
total_llh_record=np.zeros(iterations)

improvement=1.0
iteration=0

while iteration<1000 and (improvement>0.0001 or improvement<0):
    print('we are at iteration')
    print(iteration)

    iter_t0 = time.perf_counter() 
    Nk = np.sum(r_ik, axis=0) 

    piprob[iteration, :] = Nk/N 

    mu[iteration, :, :] = (np.transpose(continuousdata) @ r_ik).T / Nk[:, np.newaxis] 

    diff = continuousdata[:, np.newaxis, :] - mu[iteration][np.newaxis, :, :] 
    Sigma[iteration, :, :, :] = np.einsum('ik,ikp,ikq->kpq', r_ik, diff, diff) / Nk[:, np.newaxis, np.newaxis] 

    for j in range(p): 
        dj = categorical_columns_dims[j] 
        onehot_j = (categorical_int_data[:, j][:, np.newaxis] == np.arange(dj)[np.newaxis, :]).astype(float) 
        Psi[iteration, :, j, :dj] = (np.transpose(r_ik) @ onehot_j) / Nk[:, np.newaxis] 


    total_log_likelihood, log_likelihood_per_sample, llh=computelikelihood(mu=mu[iteration,:],Sigma=Sigma[iteration,:],Psi=Psi[iteration,:],pi=piprob[iteration,:])

    total_llh_record[iteration]=total_log_likelihood

    if iteration>=10:
        improvement=total_llh_record[iteration]-total_llh_record[iteration-1]

    em_compute_elapsed += time.perf_counter() - iter_t0 

    print('total_log_likelihood')
    print(total_log_likelihood)

    print('improvement')
    print(improvement)

    print('predicted hard clusters')
    print(Counter(np.argmax(r_ik, axis=1)))

    print('true clusters')
    print(Counter(fulldataframe['clusterlabel']))

    iter_t1 = time.perf_counter() 
    r_ik = np.exp(llh - log_likelihood_per_sample[:, np.newaxis])
    em_compute_elapsed += time.perf_counter() - iter_t1 

    outputmu = open('mu'+filenamesuffix+'.pkl', 'wb') 
    pickle.dump(mu, outputmu)
    outputmu.close()

    outputr_ik = open('r_ik'+filenamesuffix+'.pkl', 'wb') 
    pickle.dump(r_ik, outputr_ik)
    outputr_ik.close()

    outputPsi = open('Psi'+filenamesuffix+'.pkl', 'wb') 
    pickle.dump(Psi, outputPsi)
    outputPsi.close()

    outputSigma = open('Sigma'+filenamesuffix+'.pkl', 'wb') 
    pickle.dump(Sigma, outputSigma)
    outputSigma.close()

    outputpiprob = open('piprob'+filenamesuffix+'.pkl', 'wb') 
    pickle.dump(piprob, outputpiprob)
    outputpiprob.close()

    outputllh = open('llh'+filenamesuffix+'.pkl', 'wb') 
    pickle.dump(llh, outputllh) 
    outputllh.close() 

    outputiteration = open('iteration'+filenamesuffix+'.pkl', 'wb') 
    pickle.dump(iteration, outputiteration)
    outputiteration.close()

    timerecord[iteration]=em_compute_elapsed 

    iteration=iteration+1

end_time = time.time()
elapsed_time = end_time - start_time

print('elapsed_time')
print(elapsed_time)

outputelapsed_time = open('elapsed_time'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(elapsed_time, outputelapsed_time)
outputelapsed_time.close()

outputtimerecord = open('timerecord'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(timerecord, outputtimerecord) 
outputtimerecord.close() 

from scipy.optimize import linear_sum_assignment

def error_permuted(truevals, estimatedvals):

    distance_matrix = np.zeros((Ktrue, K))

    for i in range(0,Ktrue):
        for j in range(0,K):

            distance_matrix[i, j] = np.mean( np.abs(truevals[i,:] - estimatedvals[j,:]) )


    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    matched_true_means = truevals[row_ind,:]
    matched_estimated_means = estimatedvals[col_ind,:]

    mean_error = np.mean( np.abs(matched_true_means - matched_estimated_means) )

    print(f"Matched True Means:\n{matched_true_means}")
    print(f"Matched Estimated Means:\n{matched_estimated_means}")

    print(f"Mean Error: {mean_error:.10f}")

    print(f"true_ind: {row_ind}")
    print(f"estimated_ind: {col_ind}")

    return mean_error, row_ind, col_ind, matched_true_means, matched_estimated_means


def error_nopermute(truevals, estimatedvals, row_ind, col_ind):

    matched_true_means = truevals[row_ind,:]
    matched_estimated_means = estimatedvals[col_ind,:]

    mean_error = np.mean( np.abs(matched_true_means - matched_estimated_means) )

    print(f"Matched True Means:\n{matched_true_means}")
    print(f"Matched Estimated Means:\n{matched_estimated_means}")

    print(f"Mean Error: {mean_error:.10f}")

    print(f"true_ind: {row_ind}")
    print(f"estimated_ind: {col_ind}")

    return mean_error, row_ind, col_ind, matched_true_means, matched_estimated_means


lastiter=iteration-1

print('mean in the original scale')

error_permuted(truemu_kg[:,0,:], scaler.inverse_transform(mu[lastiter,:]) )

print('mean in the standardised scale')

mu_error, mu_row_ind, mu_col_ind, matched_true_mu, matched_estimated_mu=error_permuted(scaler.transform(truemu_kg[:,0,:]), mu[lastiter,:])

print('covariance in the original scale')
estimatedcovogscale=np.array([ np.diag(scaler.scale_) @ Sigma[lastiter,:] @ np.diag(scaler.scale_) for k in range(K)])

error_permuted(trueSigma_k[:,0,:], estimatedcovogscale )

print('covariance in the standardised scale')
trueSigma_kscaled=np.array([ np.diag(1/scaler.scale_) @ trueSigma_k[k, 0, :] @ np.diag(1/scaler.scale_) for k in range(Ktrue)])

sigma_error, sigma_row_ind, sigma_col_ind, matched_true_sigma, matched_estimated_sigma=error_nopermute(trueSigma_kscaled, Sigma[lastiter,:], mu_row_ind, mu_col_ind) 

def most_frequent_per_column(arr):
    n_rows, n_cols = arr.shape
    most_frequent = np.zeros(n_cols, dtype=int)  

    for col in range(n_cols):
        col_values = arr[:, col]
        unique_values, counts = np.unique(col_values, return_counts=True)
        most_frequent[col] = unique_values[np.argmax(counts)]  


    return most_frequent


from scipy.stats import mode

mapping = dict(zip(mu_col_ind, mu_row_ind))

predictedclusters=np.argmax(r_ik, axis=1) 

orderedclusters=np.vectorize(lambda x: mapping.get(x, -1))(predictedclusters)

print(np.sum(orderedclusters == component)/len(orderedclusters) )

hardclustermatchesprob=np.sum(orderedclusters == component)/len(orderedclusters)

predpsiprob=np.zeros((Ktrue,p,d_1))

Psi_error=np.zeros(p)

for j in range(0,p):
    print('we are at column '+str(j))

    print('truepsi')
    print( truepsi[:,j,:] )
    print('rearranged Psiparam prob')
    print( Psi[lastiter,mu_col_ind,j,:])

    print('error')
    Psi_error[j]=np.mean( np.abs(truepsi[:,j,:]-Psi[lastiter,mu_col_ind,j,:] ) )

    print(Psi_error[j])


print('Psi mse averaged over p columns')
Psi_error_mean=np.mean(Psi_error)
print(Psi_error_mean)

estimateclusterprob=piprob[lastiter,mu_col_ind] 

print(trueclusterprob)
print(np.abs(trueclusterprob-estimateclusterprob))

pi_error=np.mean(np.abs(trueclusterprob-estimateclusterprob)) 

print(pi_error)

mu_error_vec=mu_error
mu_row_ind_vec=mu_row_ind
mu_col_ind_vec=mu_col_ind

sigma_error_vec=sigma_error
sigma_row_ind_vec=sigma_row_ind
sigma_col_ind_vec=sigma_col_ind

outputmu_error_vec=open('mu_error_vec'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(mu_error_vec, outputmu_error_vec)
outputmu_error_vec.close()

outputmu_row_ind_vec=open('mu_row_ind_vec'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(mu_row_ind_vec, outputmu_row_ind_vec)
outputmu_row_ind_vec.close()

outputmu_col_ind_vec=open('mu_col_ind_vec'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(mu_col_ind_vec, outputmu_col_ind_vec)
outputmu_col_ind_vec.close()

outputsigma_error_vec=open('sigma_error_vec'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(sigma_error_vec, outputsigma_error_vec)
outputsigma_error_vec.close()

outputsigma_row_ind_vec=open('sigma_row_ind_vec'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(sigma_row_ind_vec, outputsigma_row_ind_vec)
outputsigma_row_ind_vec.close()

outputsigma_col_ind_vec=open('sigma_col_ind_vec'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(sigma_col_ind_vec, outputsigma_col_ind_vec)
outputsigma_col_ind_vec.close()

outputhardclustermatchesprob=open('hardclustermatchesprob'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(hardclustermatchesprob, outputhardclustermatchesprob)
outputhardclustermatchesprob.close()

outputPsi_error_mean=open('Psi_error_mean'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(Psi_error_mean, outputPsi_error_mean)
outputPsi_error_mean.close()

outputpi_error=open('pi_error'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(pi_error, outputpi_error)
outputpi_error.close()

from sklearn.covariance import LedoitWolf 
from scipy.stats import multivariate_normal 

n_bootstrap = 200      

boot_rng = np.random.default_rng(seed=987654321) 

from scipy.stats import chi2 
from scipy.stats import f as fdist 
from scipy.linalg import solve_triangular 

uq_boot_maxit = 1000   
uq_boot_tol   = 0.0001   

pi_hat    = piprob[lastiter, :].copy()          
mu_hat    = mu[lastiter, :, :].copy()           
Sigma_hat = Sigma[lastiter, :, :, :].copy()     
Psi_hat   = Psi[lastiter, :, :, :].copy()       

d = d_1 
trueclusterprobfull = np.zeros(K) 
trueclusterprobfull[mu_col_ind] = trueclusterprob 
truemu_kgfull = np.zeros((K, q)) 
truemu_kgfull[mu_col_ind, :] = scaler.transform(truemu_kg[:, 0, :]) 
trueSigma_full = np.zeros((K, q, q)) 
trueSigma_full[mu_col_ind, :] = trueSigma_kscaled 
truepsi_full = np.zeros((K, p, d)) 
for j in range(0, p): 
    truepsi_full[mu_col_ind, j, :] = truepsi[:, j, :] 


def emuq_generate_bootstrap(pi_e, mu_e, Sigma_e, Psi_e, rng): 

    z_b = rng.choice(K, size=N, p=pi_e/np.sum(pi_e)) 
    cont_b = np.zeros((N, q)) 
    cat_b = np.zeros((N, p), dtype=int) 
    for k in range(0, K): 
        idx = np.where(z_b == k)[0] 
        if idx.size == 0: 
            continue 
        cont_b[idx, :] = rng.multivariate_normal(mean=mu_e[k, :], cov=Sigma_e[k, :], size=idx.size) 
        for j in range(0, p): 
            dj = categorical_columns_dims[j] 
            probs = Psi_e[k, j, :dj]; probs = probs/np.sum(probs) 
            cat_b[idx, j] = rng.choice(dj, size=idx.size, p=probs) 


    return cont_b, cat_b 


def emuq_fit_em(cont_b, cat_int_b, pi0, mu0, Sigma0, Psi0, maxit=uq_boot_maxit, tol=uq_boot_tol): 

    Nb = cont_b.shape[0] 
    def estep(pi_e, mu_e, Sigma_e, Psi_e): 
        gauss = np.zeros((Nb, K)) 
        for k in range(0, K): 
            gauss[:, k] = multivariate_normal.logpdf(cont_b, mean=mu_e[k, :], cov=Sigma_e[k, :], allow_singular=True) 


        pll = np.zeros((Nb, K)) 
        for j in range(0, p): 

            pll = pll + np.log(np.maximum(Psi_e[:, j, cat_int_b[:, j]], 1e-300)).T 


        llh_e = np.log(pi_e)[np.newaxis, :] + gauss + pll 
        lps = logsumexp(llh_e, axis=1) 
        return np.exp(llh_e - lps[:, np.newaxis]), np.sum(lps) 


    r_b, tll_prev = estep(pi0, mu0, Sigma0, Psi0) 
    pi_c, mu_c, Sigma_c, Psi_c = pi0.copy(), mu0.copy(), Sigma0.copy(), Psi0.copy() 
    for it in range(0, maxit): 
        Nk_b = np.sum(r_b, axis=0) 
        pi_c = Nk_b/Nb 
        mu_c = (np.transpose(cont_b) @ r_b).T / Nk_b[:, np.newaxis] 
        dfb = cont_b[:, np.newaxis, :] - mu_c[np.newaxis, :, :] 
        Sigma_c = np.einsum('ik,ikp,ikq->kpq', r_b, dfb, dfb) / Nk_b[:, np.newaxis, np.newaxis] 
        Psi_c = np.zeros_like(Psi0) 
        for j in range(0, p): 
            dj = categorical_columns_dims[j] 
            oh = (cat_int_b[:, j][:, np.newaxis] == np.arange(dj)[np.newaxis, :]).astype(float) 
            Psi_c[:, j, :dj] = (np.transpose(r_b) @ oh) / Nk_b[:, np.newaxis] 


        r_b, tll = estep(pi_c, mu_c, Sigma_c, Psi_c) 
        if it >= 1 and abs(tll - tll_prev) < tol: 
            break 
        tll_prev = tll 


    return pi_c, mu_c, Sigma_c, Psi_c, it+1 


def emuq_align(mu_b, mu_ref): 

    C = np.zeros((K, K)) 
    for i in range(0, K): 
        for jj in range(0, K): 
            C[i, jj] = np.mean(np.abs(mu_ref[i, :] - mu_b[jj, :])) 


    row_ind_a, col_ind_a = linear_sum_assignment(C) 
    perm = np.zeros(K, dtype=int) 
    perm[row_ind_a] = col_ind_a 
    return perm 


print('running EM parametric bootstrap for uncertainty quantification') 
boot_pi = np.zeros((n_bootstrap, K)) 
boot_mu = np.zeros((n_bootstrap, K, q)) 
boot_Sigma = np.zeros((n_bootstrap, K, q, q)) 
boot_Psi = np.zeros((n_bootstrap, K, p, Psi_hat.shape[2])) 

emuq_boot_iters = np.zeros(n_bootstrap, dtype=int) 

timerecord2 = np.zeros(n_bootstrap) 
emuq_gen_elapsed = 0.0 
emuq_start_time = time.time() 
for b in range(0, n_bootstrap): 
    boot_t0 = time.perf_counter() 
    cont_b, cat_b = emuq_generate_bootstrap(pi_hat, mu_hat, Sigma_hat, Psi_hat, boot_rng) 
    pi_b, mu_b, Sigma_b, Psi_b, niter_b = emuq_fit_em(cont_b, cat_b, pi_hat, mu_hat, Sigma_hat, Psi_hat) 
    emuq_boot_iters[b] = niter_b 
    perm = emuq_align(mu_b, mu_hat) 
    boot_pi[b] = pi_b[perm] 
    boot_mu[b] = mu_b[perm] 
    boot_Sigma[b] = Sigma_b[perm] 
    boot_Psi[b] = Psi_b[perm] 
    emuq_gen_elapsed += time.perf_counter() - boot_t0 
    timerecord2[b] = emuq_gen_elapsed 
    if (b+1) % 50 == 0: 
        print('  bootstrap replicate '+str(b+1)+'/'+str(n_bootstrap)) 


emuq_elapsed = time.time() - emuq_start_time 

print('EM bootstrap refits done in '+str(emuq_elapsed)+'s'+ 
      '  (warm-started EM iterations per refit: mean '+str(round(float(np.mean(emuq_boot_iters)),1))+ 
      ', max '+str(int(np.max(emuq_boot_iters)))+')') 

uq_tri = np.triu_indices(q) 

def emuq_psi_vec(PSI_k): 
    return np.concatenate([PSI_k[j, :categorical_columns_dims[j]] for j in range(0, p)]) 


boot_mu_k = [boot_mu[:, k, :] for k in range(0, K)] 
boot_Sigma_k = [boot_Sigma[:, k][:, uq_tri[0], uq_tri[1]] for k in range(0, K)] 
boot_Psi_k = [np.stack([emuq_psi_vec(boot_Psi[b, k]) for b in range(0, n_bootstrap)]) for k in range(0, K)] 
boot_pi_all = boot_pi 
boot_mu_all = boot_mu.reshape(n_bootstrap, K*q) 
boot_Sigma_all = np.concatenate(boot_Sigma_k, axis=1) 
boot_Psi_all = np.concatenate(boot_Psi_k, axis=1) 
boot_joint = np.concatenate([boot_pi_all, boot_mu_all, boot_Sigma_all, boot_Psi_all], axis=1) 

pi_hat_v = pi_hat 
mu_hat_v = mu_hat.reshape(K*q) 
Sigma_hat_v = np.concatenate([Sigma_hat[k][uq_tri] for k in range(0, K)]) 
Psi_hat_v = np.concatenate([emuq_psi_vec(Psi_hat[k]) for k in range(0, K)]) 
joint_hat_v = np.concatenate([pi_hat_v, mu_hat_v, Sigma_hat_v, Psi_hat_v]) 

pi_true_v = trueclusterprobfull 
mu_true_v = truemu_kgfull.reshape(K*q) 
Sigma_true_v = np.concatenate([trueSigma_full[k][uq_tri] for k in range(0, K)]) 
Psi_true_v = np.concatenate([emuq_psi_vec(truepsi_full[k]) for k in range(0, K)]) 
joint_true_v = np.concatenate([pi_true_v, mu_true_v, Sigma_true_v, Psi_true_v]) 

def emuq_coverage(replicates, center, truevalue): 

    replicates = np.atleast_2d(replicates) 
    dim = replicates.shape[1] 

    Bn = replicates.shape[0] 
    Su = np.atleast_2d(np.cov(replicates, rowvar=False)) 
    evals, evecs = np.linalg.eigh(Su) 
    keep = evals > max(evals.max(), 0.0) * dim * np.finfo(float).eps 
    r = int(np.sum(keep)) 
    if r == 0 or r >= Bn: 
        raise ValueError('emuq_coverage: block rank r=' + str(r) + ' must satisfy 0 < r < B=' + str(Bn) + '; increase n_bootstrap') 
    Ur, lam = evecs[:, keep], evals[keep] 
    x = np.atleast_1d(truevalue) - np.atleast_1d(center) 
    y = Ur.T @ x 
    in_support = np.linalg.norm(x - Ur @ y) <= 1e-8 * max(np.linalg.norm(x), 1e-300) 
    cutoff = r * (Bn - 1) / (Bn - r) * fdist.ppf(0.95, r, Bn - r) 
    lognorm = -0.5 * r * np.log(2.0 * np.pi) - 0.5 * np.sum(np.log(lam)) 
    threshold = lognorm - 0.5 * cutoff 
    logdens_true = float(lognorm - 0.5 * np.sum(y ** 2 / lam)) if in_support else -np.inf 
    return float(threshold), logdens_true, bool(logdens_true > threshold) 


emuq_boundary_tol = 1e-10 

def emuq_zero_variance_coords(replicates): 

    return np.where(np.std(np.atleast_2d(replicates), axis=0) <= emuq_boundary_tol)[0] 


thr_joint, ltrue_joint, cov_joint = emuq_coverage(boot_joint,     joint_hat_v, joint_true_v) 
thr_pi,    ltrue_pi,    cov_pi    = emuq_coverage(boot_pi_all,    pi_hat_v,    pi_true_v) 
thr_Sigma, ltrue_Sigma, cov_Sigma = emuq_coverage(boot_Sigma_all, Sigma_hat_v, Sigma_true_v) 
thr_mu,    ltrue_mu,    cov_mu    = emuq_coverage(boot_mu_all,    mu_hat_v,    mu_true_v) 
thr_Psi,   ltrue_Psi,   cov_Psi   = emuq_coverage(boot_Psi_all,   Psi_hat_v,   Psi_true_v) 

logposteriorallEM     = [thr_joint, thr_pi, thr_Sigma, thr_mu, thr_Psi]        
logposteriortrueallEM = [ltrue_joint, ltrue_pi, ltrue_Sigma, ltrue_mu, ltrue_Psi] 
coverageresultsEM     = [cov_joint, cov_pi, cov_Sigma, cov_mu, cov_Psi]        

print('EM bootstrap coverage (joint, pi, Sigma, mu, Psi):') 
print('  region thresholds :', logposteriorallEM) 
print('  true log-densities:', logposteriortrueallEM) 
print('  covered           :', coverageresultsEM) 

Sigma_perk_thrEM = np.zeros(K); Sigma_perk_trueEM = np.zeros(K); Sigma_perk_covEM = [] 
mu_perk_thrEM    = np.zeros(K); mu_perk_trueEM    = np.zeros(K); mu_perk_covEM    = [] 
Psi_perk_thrEM   = np.zeros(K); Psi_perk_trueEM   = np.zeros(K); Psi_perk_covEM   = [] 
for k in range(0, K): 
    t, lt, c = emuq_coverage(boot_Sigma_k[k], Sigma_hat[k][uq_tri], trueSigma_full[k][uq_tri]) 
    Sigma_perk_thrEM[k] = t; Sigma_perk_trueEM[k] = lt; Sigma_perk_covEM.append(c) 
    t, lt, c = emuq_coverage(boot_mu_k[k], mu_hat[k, :], truemu_kgfull[k, :]) 
    mu_perk_thrEM[k] = t; mu_perk_trueEM[k] = lt; mu_perk_covEM.append(c) 
    t, lt, c = emuq_coverage(boot_Psi_k[k], emuq_psi_vec(Psi_hat[k]), emuq_psi_vec(truepsi_full[k])) 
    Psi_perk_thrEM[k] = t; Psi_perk_trueEM[k] = lt; Psi_perk_covEM.append(c) 


logposteriorall_perkEM     = [Sigma_perk_thrEM, mu_perk_thrEM, Psi_perk_thrEM] 
logposteriortrueall_perkEM = [Sigma_perk_trueEM, mu_perk_trueEM, Psi_perk_trueEM] 
coverageresults_perkEM     = [Sigma_perk_covEM, mu_perk_covEM, Psi_perk_covEM] 

print('per-k Sigma coverage:', coverageresults_perkEM[0]) 
print('per-k mu    coverage:', coverageresults_perkEM[1]) 
print('per-k Psi   coverage:', coverageresults_perkEM[2]) 

coverageflagsEM = [len(emuq_zero_variance_coords(X)) > 0 for X in 
                   [boot_joint, boot_pi_all, boot_Sigma_all, boot_mu_all, boot_Psi_all]] 
coverageflags_perkEM = [[len(emuq_zero_variance_coords(boot_Sigma_k[k])) > 0 for k in range(0, K)], 
                        [len(emuq_zero_variance_coords(boot_mu_k[k])) > 0 for k in range(0, K)], 
                        [len(emuq_zero_variance_coords(boot_Psi_k[k])) > 0 for k in range(0, K)]] 

boundaryreportEM = {'tol': emuq_boundary_tol, 
                    'pi': [(k, float(pi_hat[k])) for k in range(0, K) if np.std(boot_pi[:, k]) <= emuq_boundary_tol], 
                    'Psi': [(k, j, g, float(Psi_hat[k, j, g])) for k in range(0, K) for j in range(0, p) 
                            for g in range(0, categorical_columns_dims[j]) if np.std(boot_Psi[:, k, j, g]) <= emuq_boundary_tol]} 
if any(coverageflagsEM) or any(any(f) for f in coverageflags_perkEM): 
    print('WARNING: boundary estimates found; the flagged coverage results below are unreliable (parametric bootstrap') 
    print('         is inconsistent at the boundary). Non-varying pi_k (k, pi_hat):', boundaryreportEM['pi'], 
          '| Psi_kjg (k, j, g, Psi_hat):', boundaryreportEM['Psi']) 
print('flags (joint, pi, Sigma, mu, Psi):', coverageflagsEM) 
print('per-k flags [Sigma], [mu], [Psi]  :', coverageflags_perkEM) 

emuq_true_of_EM = {int(c): int(r) for r, c in zip(mu_row_ind, mu_col_ind)} 
emuq_agg_keys = ['joint', 'pi', 'Sigma_allk', 'mu_allk', 'Psi_allk'] 

def emuq_plain(v): 
    return bool(v) if isinstance(v, (bool, np.bool_)) else float(v) 


def emuq_label_agg(vals): 
    return {key: emuq_plain(v) for key, v in zip(emuq_agg_keys, vals)} 


def emuq_label_perk(rows): 
    out = {} 
    for blk, row in zip(['Sigma', 'mu', 'Psi'], rows): 
        for k in sorted(range(0, K), key=lambda kk: emuq_true_of_EM.get(kk, K + kk)): 
            name = blk + '_truek' + str(emuq_true_of_EM[k]) if k in emuq_true_of_EM else blk + '_unmatchedEMk' + str(k) 
            out[name] = emuq_plain(row[k]) 


    return out 


coverageresultsEM          = emuq_label_agg(coverageresultsEM) 
logposteriorallEM          = emuq_label_agg(logposteriorallEM) 
logposteriortrueallEM      = emuq_label_agg(logposteriortrueallEM) 
coverageflagsEM            = emuq_label_agg(coverageflagsEM) 
coverageresults_perkEM     = emuq_label_perk(coverageresults_perkEM) 
logposteriorall_perkEM     = emuq_label_perk(logposteriorall_perkEM) 
logposteriortrueall_perkEM = emuq_label_perk(logposteriortrueall_perkEM) 
coverageflags_perkEM       = emuq_label_perk(coverageflags_perkEM) 

boundaryreportEM = {'tol': boundaryreportEM['tol'], 
                    'EM_to_true_component': emuq_true_of_EM, 
                    'pi': [{'EM_component': k, 'true_component': emuq_true_of_EM.get(k), 'pi_hat': v} 
                           for (k, v) in boundaryreportEM['pi']], 
                    'Psi': [{'EM_component': k, 'true_component': emuq_true_of_EM.get(k), 'j': j, 'g': g, 'Psi_hat': v} 
                            for (k, j, g, v) in boundaryreportEM['Psi']]} 

print('labelled coverage (aggregate) :', coverageresultsEM) 
print('labelled coverage (per true k):', coverageresults_perkEM) 
print('labelled boundary flags       :', coverageflagsEM, coverageflags_perkEM) 

outputcoverageresultsEM = open('coverageresultsEM'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(coverageresultsEM, outputcoverageresultsEM) 
outputcoverageresultsEM.close() 

outputlogposteriorallEM = open('logposteriorallEM'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(logposteriorallEM, outputlogposteriorallEM) 
outputlogposteriorallEM.close() 

outputlogposteriortrueallEM = open('logposteriortrueallEM'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(logposteriortrueallEM, outputlogposteriortrueallEM) 
outputlogposteriortrueallEM.close() 

outputcoverageresults_perkEM = open('coverageresults_perkEM'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(coverageresults_perkEM, outputcoverageresults_perkEM) 
outputcoverageresults_perkEM.close() 

outputlogposteriorall_perkEM = open('logposteriorall_perkEM'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(logposteriorall_perkEM, outputlogposteriorall_perkEM) 
outputlogposteriorall_perkEM.close() 

outputlogposteriortrueall_perkEM = open('logposteriortrueall_perkEM'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(logposteriortrueall_perkEM, outputlogposteriortrueall_perkEM) 
outputlogposteriortrueall_perkEM.close() 

outputcoverageflagsEM = open('coverageflagsEM'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(coverageflagsEM, outputcoverageflagsEM) 
outputcoverageflagsEM.close() 

outputcoverageflags_perkEM = open('coverageflags_perkEM'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(coverageflags_perkEM, outputcoverageflags_perkEM) 
outputcoverageflags_perkEM.close() 

outputboundaryreportEM = open('boundaryreportEM'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(boundaryreportEM, outputboundaryreportEM) 
outputboundaryreportEM.close() 

outputboot_estimatesEM = open('boot_estimatesEM'+filenamesuffix+'.pkl', 'wb') 
pickle.dump({'pi': boot_pi, 'mu': boot_mu, 'Sigma': boot_Sigma, 'Psi': boot_Psi}, outputboot_estimatesEM) 
outputboot_estimatesEM.close() 

outputtimerecord2 = open('boottimerecord'+filenamesuffix+'.pkl', 'wb') 
pickle.dump(timerecord2, outputtimerecord2) 
outputtimerecord2.close() 

print('bootstrap-sample generation time (timerecord2): total '+str(round(float(timerecord2[-1]),3))+'s for '+str(n_bootstrap)+' replicates; main EM loop compute time (timerecord): '+str(round(float(timerecord[lastiter]),3))+'s -> additional cost factor ~'+str(round(float(timerecord2[-1])/max(float(timerecord[lastiter]),1e-9),1))+'x the single EM fit') 

print('done EM parametric bootstrap UQ') 

print('done all') 

import itertools
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import copy
import random
import paddy
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1.25

pk_a_25 = np.load("pk_a_25.npy")
pk_a_1 = np.load("pk_a_1.npy")


import random

random.seed(1)
np.random.seed(1)

def pk_bd(inp):
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(np.abs(inp))
    inp = scaler.transform(inp)
    org = []
    org.append(min(inp[:,0]))
    org.append(max(inp[:,1]))
    org.append(max(inp[:,2]))
    org.append(max(inp[:,3]))
    org.append(max(inp[:,4]))
    org.append(max(inp[:,5]))
    org.append(max(inp[:,6]))
    org.append(min(inp[:,7]))
    org.append(min(inp[:,8]))
    org.append(min(inp[:,9]))
    dist =[]
    for i in inp:
        dist.append(np.linalg.norm(i - org))
    return(dist)

ain = np.abs(np.reshape(np.array([pk_a_25,pk_a_1]),(-1,10)))
print(ain)

dist=pk_bd(ain)

sort_dist = np.argsort(dist)
print(sort_dist)


#pk_a shape is 180 by 10 (anal. vals), first dimension is the six tags repeated
#for each of the five biotin-IDs which in turn
#is repeated for each six formats
# so it would go six tags for biotin 1 then six tags for biotin 2 ... for the frist format and repeated six times

tags = ["Generic","6","8","9","12","17"]
biotin = ["6","8","9","12","17"]
ant = ['0.25','1.0']
form = ['1','2','3','4','5','6']
pl = []
pl2 = []
for i in sort_dist:
    a = i//180 #1 for pk_a_1 and 0 for pk_a_25
    g = i + 1 - (180*a)
    f = g%30 # if the last it is 0
    F = g//30 - (0 ** f)
    b = g%6 # if the last it is 0
    B = (g - (F*30))//6  - (0**b)
    T = g%6
    pl.append(tags[T]+"-"+biotin[B]+ ", ["+ant[a]+"], "+form[F])
    pl2.append([T,B,a,F])



for i in range(len(pl)):
    print(pl[i] + " | distance:" + str(dist[sort_dist[i]]) + " | " + str(sort_dist[i]))


observations = [[],[],[],[],[]]
for i in range(len(pl2)):
    observations[0].append(pl2[i][0])
    observations[1].append(pl2[i][1])
    observations[2].append(pl2[i][2])
    observations[3].append(pl2[i][3])
    observations[4].append(sort_dist[i])#replace with index, use index later with ain and pk_db()


print(observations)

print(observations[0])
print(observations[1])
print(observations[2])
print(observations[3])
print("obs4",observations[4])


class Numeric_Assistant(object):
    """
    Sampling assistant that manages numeric space.
    Repeats if false will remove observations after
    a set of observations

    Parameters
    ----------
    observations : array_like, shape(N+1,M)
        A rank-2 array of M observations where rows contain
        the cordinates in N-dimensional parameter space, with
        the last row being the response value.

    repeats : bool (default: True)
        Boolean to determine if observations can be repeatedly
        sampled.

    raw_responce : bool (default: False)
        Boolean to determine if the responce is normalized
        when an observation is sampled.


    Attributes
    ----------
    obs_clone : array_like, shape(,2)
        Array that is initially a clone of the observation
        parameter, and has observations removed by the 
        `obs_cloner` method.

    norm_clone : array_like, shape(,2)


    Methods
    ------- 
    sample(inputs)

    norm_cloner()
        Clones 

    obs_cloner()
        Uses `black_list` to ommit observations before
        cloning to `obs_clone`.

    min_max(x, minV, maxV)
        Used for normalizing rows in `observations` or 
        `obs_clone` if `repeats` is False.

    """
    def __init__(self,observations,repeats=True,raw_responce=False):
        self.observations = observations
        self.repeats = repeats
        self.obs_clone = np.array(copy.deepcopy(observations))
        self.black_list = []
        self.norm_cloner()
        self.raw_responce = raw_responce

    def sample(self,inputs):
        temp = []
        for i in inputs:
            temp.append(i[0])
        inputs = temp
        distance_list = []
        distance_vectors = []
        c = 0
        for i in range(len(self.obs_clone[0])):#get distances for every param combo
            distance_vectors.append(np.linalg.norm(self.norm_clone[:-1,c] - inputs))
            c += 1
        distance_list.append(distance_vectors)#append distances for each domain/param
        score_list = []
        if not self.repeats:#doesnt have error handling for equal distances
            self.black_list = []
            #print("distance list:",distance_list)
            #print("norm clone", self.norm_clone)
            #print("distance list len:",len(distance_list[0]))
            if len(distance_list[0]) > 1:
                for i in distance_list:
                    repeats = i.count(min(i))
                    if repeats == 1:
                        if self.raw_responce:
                            print(self.obs_clone[:-1,np.argmin(i)])
                            score_list.append(self.obs_clone[-1][np.argmin(i)])
                        else:
                            score_list.append(self.norm_clone[-1][np.argmin(i)])
                        self.black_list.append(np.argmin(i))
                    if repeats > 1:
                        d = 0
                        r_list = []
                        for j in i:
                            if j == min(i):
                                r_list.append(d)
                            d +=1
                        rc = random.choice(r_list)
                        if self.raw_responce:
                            score_list.append(self.obs_clone[-1][rc])
                        else:
                            score_list.append(self.norm_clone[-1][rc])
                        self.black_list.append(rc)
            if len(distance_list[0]) == 1:
                if self.raw_responce:
                    score_list.append(self.obs_clone[-1][0])
                else:
                    score_list.append(self.norm_clone[-1][0])
                self.black_list.append(0)

            self.norm_cloner()#renormalizes after sampling and removes observations
            #distance_list is values of selection out of vector of posibilites
            #print("Black list len:", len(self.black_list))
            #print("# Observations:", len(self.obs_clone[0]))
        else:
            for i in distance_list:
                if self.raw_responce:
                    score_list.append(self.obs_clone[-1][np.argmin(i)])
                else:
                    score_list.append(self.norm_clone[-1][np.argmin(i)])
        #print(score_list)
        return(score_list)

    def min_max(self, x, minV, maxV):
        if minV != maxV:
            x = x
            minV = minV
            maxV = maxV
            return((x-minV)/(maxV-minV))
        else:
            return(1)

    def norm_cloner(self):
        #clones observations with normalized values between 0 and 1
        self.norm_clone =[]
        if self.repeats:
            for row in self.observations:
                minV, maxV = min(row), max(row)
                temp =[]
                for value in row:
                    normed = self.min_max(value, minV, maxV)
                    temp.append(normed)
                self.norm_clone.append(temp)
        else: 
            self.obs_cloner()
            if len(self.obs_clone[0]) > 0: 
                for row in self.obs_clone:
                    minV, maxV = min(row), max(row)
                    temp =[]
                    for value in row:
                        normed = self.min_max(value, minV, maxV)
                        temp.append(normed)
                    self.norm_clone.append(temp)
        self.norm_clone = np.array(self.norm_clone)    

    def obs_cloner(self):
        #uses black list to ommit prior observations
        obs_clone2 = []
        c2 = 0
        for i in self.obs_clone:
            obs_clone2.append([])
            c = 0
            for j in i:
                if c in self.black_list:
                    pass
                else:
                    obs_clone2[c2].append(j)
                c += 1
            c2 += 1
        self.obs_clone = np.array(obs_clone2)

assistant = Numeric_Assistant(observations=observations,repeats=False)


obs = [[0,1,2,3,4,5,2],[-1,-.5,3,2,11,3,-2],[12,1,4,2,6,5,1.2]]


norm_param = paddy.PaddyParameter(param_range=[0,1,.01],
                                param_type='continuous',
                                limits=[0,1], gaussian='default',
                                normalization = False)

class PK_space(object):
    def __init__(self):
        self.a = norm_param
        self.b = norm_param
        self.c = norm_param
        self.d = norm_param

pks = PK_space()


def min_max( x, minV, maxV):
    #print(x,minV,maxV)
    if minV != maxV:
        x = x
        minV = minV[0]
        maxV = maxV[0]
        return([(x-minV)/(maxV-minV)])
    else:
        return(1)

def inv_min_max( x, minV, maxV):
    #print(x,minV,maxV)
    if minV != maxV:
        x = x
        minV = minV[0]
        maxV = maxV[0]
        return((x*(maxV-minV))+minV)
   


observations = np.array(observations)

assistant = Numeric_Assistant(observations=observations,repeats=False,raw_responce=True)
#s = [[1,.3,.4,.7]]
#assistant.sample(s)

sampled_points = []

'''
def sampler(inputs):
    out = assistant.sample(inputs)
    sampled_points.append(out)
    if len(out) > 0:
        if len(sampled_points) > 1:
            out = min_max(out[0],min(sampled_points),max(sampled_points))
        return(-out[0])
    else:
        return(-10000000)
'''      

def sampler(inputs):
    out = assistant.sample(inputs)
    #print("out",out)
    if len(out) != 0:
        sampled_points.append(out[0])
    else:
        print("BAD")
        out = -1000
    return(out)


runner = paddy.PFARunner(space=pks,
                        eval_func=sampler,
                        paddy_type='generational',
                        rand_seed_number=20,
                        yt=10,
                        Qmax=5,
                        r=.2,
                        iterations=20)

all_list = []
p_list = [5,10,15,20]
for yt in p_list:
    for qmax in p_list:
        z_list = []
        answer_c = []
        paddy_list = []
        seed_c = []
        for z in range(100):
            DONE = False
            while not DONE:
                sampled_points = []
                norm_param = paddy.PaddyParameter(param_range=[0,1,.01],
                                                param_type='continuous',
                                                limits=[0,1], gaussian='default',
                                                normalization = False)
                class PK_space(object):
                    def __init__(self):
                        self.a = norm_param
                        self.b = norm_param
                        self.c = norm_param
                        self.d = norm_param
                pks = PK_space()
                assistant = Numeric_Assistant(observations=observations,repeats=False,raw_responce=True)
                runner = paddy.PFARunner(space=pks,
                                        eval_func=sampler,
                                        paddy_type='generational',
                                        rand_seed_number=15,
                                        yt=yt,
                                        Qmax=qmax,
                                        r=.1,
                                        iterations=1)
                try:
                    runner.run_paddy(file_name="trials/trial_{0}".format(z))
                except:
                    print('oops')
                dist_indx = []
                inital_dists = []
                for i in runner.seed_fitness:
                    # this may need to be used to fix bellow, issue with currently the wrong index is being used
                    inital_dists.append(ain[i][0])
                    dist_indx.append(i[0])
                    #print(pl2[])
                print(dist_indx)
                if 305 in dist_indx or [305] in dist_indx:
                    print(z)
                    if z ==30:
                        print('error')
                        print(DONE)
                        print(not DONE)
                    pass
                else:
                    DONE = True
            if 305 in dist_indx:
                print('this sucks')
                print(yo)

            #print("here")
            #print(runner.paddy_counter, runner.iterations)
            dists = pk_bd(np.array(inital_dists))
            #print(dists)
            #print("runner len",len(runner.seed_fitness))
            for i in range(len(runner.seed_fitness)):
                runner.seed_fitness[i] = -dists[i]
            print(runner.seed_fitness)
            print(runner.generation_data)
            runner.paddy_plot_and_print("gen_fitness")
            runner.recover_run()
            #print(runner.seed_counter)
            #print(len(runner.seed_fitness))
            dists = []
            #print(runner.generation_data)
            #print(runner.seed_fitness)
            for i in np.arange(runner.generation_data['1'][0],runner.generation_data['1'][1]):
                #print(i,ain[i])
                dist_indx.append(runner.seed_fitness[i][0])
                inital_dists.append(ain[runner.seed_fitness[i]][0])
            dists = pk_bd(np.array(inital_dists))
            #print(len(dists)) 
            #print(len(runner.seed_fitness))
            runner.seed_fitness = []
            for i in range(len(dists)):
                runner.seed_fitness.append(-dists[i])
            #print("obs",len(assistant.obs_clone[0]))
            #print(360-len(runner.seed_fitness))
            #print(dist_indx)
            c = 2
            while c < 50:
                print("min",min(dist_indx))
                runner.extend_paddy(1)
                for v in np.arange(runner.generation_data[str(c)][0],runner.generation_data[str(c)][1]):
                    print(v)
                    dist_indx.append(runner.seed_fitness[v])
                    inital_dists.append(ain[runner.seed_fitness[v]][0])
                dists = pk_bd(np.array(inital_dists))
                runner.seed_fitness = []
                for i in range(len(dists)):
                    print("a dist", dists[i])
                    runner.seed_fitness.append(-dists[i])
                c += 1
                if len(runner.seed_fitness) > 360:
                    c = 300
                    break
                mv = dist_indx[np.argmax(runner.seed_fitness)]
                print(min(dist_indx))
                print("mv",mv)
                print("paddy index",runner.paddy_counter)
                print("fitness",runner.seed_fitness)
                if mv == [305] or mv == 305:
                    print(len(np.arange(runner.generation_data[str(c-1)][0],runner.generation_data[str(c-1)][1])))
                    print(runner.seed_fitness)
                    print(dist_indx)
                    if np.argmax(runner.seed_fitness) < 5:
                        print(runner.seed_fitness)
                        print(z)
                        print(yooooo)
                    z_list.append(np.argmax(runner.seed_fitness))
                    c = 300
                    break
            
            runner.save_paddy(new_file_name='trials/trial_{0}'.format(z))
                
            #print(runner.seed_fitness)
            #print("obs",len(assistant.obs_clone[0]))
            #print(360-len(runner.seed_fitness))
            #seed_f = runner.seed_fitness
            #seed = runner.get_top_seed()[0].split("_")[1]
            #seed_s = -1*runner.seed_fitness[int(runner.get_top_seed()[0].split("_")[1])]
            #og_seed_s = inv_min_max(seed_s,min(sampled_points[:seed]),max(sampled_points[:seed]))
            #v = np.where(observations[-1] == -1*runner.seed_fitness[int(runner.get_top_seed()[0].split("_")[1])])
            v = np.array(dist_indx[np.argmin(dist_indx)]).flatten()
            seed_c.append(runner.seed_counter)
            paddy_list.append(v)
            answer_c.append(np.argmax(runner.seed_fitness))
        all_list.append(z_list)

np.save('all_sims',all_list)
print(z_list)
print(paddy_list)

label_list = []
pl_list = []
print("paddy_list",paddy_list)
print(type(paddy_list[0]))
for i in paddy_list:
    for j in i:
        print(i,j)
        print(pl[j])
        label_list.append(pl[j])
        pl_list.append(j)

data = dict()
pls = set(pl_list)
ll = set(label_list)
for i in pls:
    data[pl[i]] = pl_list.count(i)



labels = list(data.keys())
values = list(data.values())

fig,ax = plt.subplots()
ax.spines.top.set_visible(False)
ax.spines.right.set_visible(False)
plt.hist(z_list,50)
#plt.xticks(rotation=45,ha="right")
#plt.title("Paddy Solutions", fontsize = 18)
plt.ylabel("Solution Count", fontsize = 16)
plt.xlabel("Number of Evaluations", fontsize = 16)
plt.tight_layout()
plt.savefig("paddy_anti_ID_sim.pdf", dpi=300)
plt.show()



fig, ax = plt.subplots()

# Customize spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Create histogram with aesthetics
n, bins, patches = plt.hist(
    z_list,
    bins=50,
    color='skyblue',  # Aesthetic color
    edgecolor='white',  # Light outline around bars
    linewidth=0.7      # Thickness of the bar outline
)

# Customize labels and layout
plt.ylabel("Solution Count", fontsize=16)
plt.xlabel("Number of Evaluations", fontsize=16)
plt.tight_layout()

# Save and show the plot
plt.savefig("paddy_anti_ID_sim.pdf", dpi=300)
plt.show()

dl = []
pl2 = []
for i in paddy_list:
    for j in i:
        pl2.append(j)
        print(j)
        print(dist[sort_dist[j]])
        dl.append((dist[sort_dist[j]]))

dl = set(dl)
dl = list(dl)
dl.sort()
pl2 = set(pl2)
pl2 = list(pl2)
pl2.sort()

dist.sort()

x = []
for i in range(len(dist)):
    x.append(i)


fig,ax = plt.subplots()
ax.spines.top.set_visible(False)
ax.spines.right.set_visible(False)
plt.plot(x,dist)
plt.xlabel("Sorted Condition Indexes",fontsize=16)
plt.ylabel("Distance Value",fontsize=16)
plt.xticks(np.arange(0,380,40))
plt.scatter(pl2,dl,s=4.5,c='r')
plt.savefig('dis_line.pdf',dpi=300)
plt.show()


print(seed_c)
print(len(observations[0]))

print(answer_c)
print(data)
print("###############")
print(len(answer_c))
print(sum(answer_c)/100)
print(z_list)
print(min(z_list))
print(len(z_list))
#>>> np.where(sort_dist == 281)
#(array([9]),)
#>>> sort_dist[9]
#281
#>>> pl[9]
'Generic tag 58BCP biotin  antigen [1 ug] format 4'


#np.where(sort_dist == 281)
#(array([9]),)
#>>> sort_dist[9]
#281
#>>> pl[9]
[[76.17, 60.38, 77.99, 77.77], [80.34, 76.26, 89.81, 91.4], [71.55, 80.34, 92.84, 98.6], [84.23, 78.71, 101.06, 108.35]]

import matplotlib.pyplot as plt
import numpy as np
import seaborn

# Data
p_list = [5, 10, 15, 20]
data = np.array([[76.17, 60.38, 77.99, 77.77], 
                 [80.34, 76.26, 89.81, 91.4], 
                 [71.55, 80.34, 92.84, 98.6], 
                 [84.23, 78.71, 101.06, 108.35]])

# Create a heatmap
plt.figure(figsize=(8, 6))
heatmap = seaborn.heatmap(data, cmap='crest').set(xlabel='qmax', ylabel='yt')
heatmap.set(xlabel='qmax', ylabel='yt')
# Add color bar
cbar = plt.colorbar(heatmap)
cbar.set_label('Value', rotation=270, labelpad=15)


# Add labels to the heatmap
plt.xticks(ticks=np.arange(len(p_list)), labels=['5','10','15','20'])
plt.yticks(ticks=np.arange(len(p_list)), labels=['5','10','15','20'])
plt.xlabel('qmax')
plt.ylabel('yt')
plt.title('Heatmap: Low Values are Hot')

# Display the heatmap
plt.show()

#######################
import numpy as np
import matplotlib.pyplot as plt

# Load the data
z_list = np.load('all_sims.npy')[1]

# Create the figure and axis
fig, ax = plt.subplots()

# Hide the top and right spines
ax.spines.top.set_visible(False)
ax.spines.right.set_visible(False)

# Plot the histogram
bars = plt.hist(z_list, bins=25, alpha=0.9, edgecolor='black', linewidth=1.2)

# Set the bar background to a light color
for patch in bars[2]:
    patch.set_facecolor('#d8e3f5')

# Customize the axes
ax.spines.left.set_linewidth(5.5)
ax.spines.bottom.set_linewidth(5.5)
ax.tick_params(axis='both', which='major', labelsize=20, width=4.5, length=6)
ax.tick_params(axis='both', which='minor', width=3.5, length=4)
ax.set_yticklabels([0,2,4,6,8,10,12], fontsize=20)
# Add labels
#plt.ylabel("Solution Count", fontsize=34)
#plt.xlabel("Number of Experiments", fontsize=34)

# Optimize layout
plt.tight_layout()

# Save the figure
plt.savefig("paddy_anti_ID_sim.pdf", dpi=300)

# Display the plot
plt.show()

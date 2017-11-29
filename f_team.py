import numpy as np

#f_team calculates the fitness function for a team.
#INPUTS:
#soc_network - a social network of preferences in form of a list of edges and weights
#n - a number of students in class
#team - indices of the team members
#scores - the scores of the class
#wplus - weighting the positive relationships
#wminus - weighting the negative relationships
#wneutr - weighting the neutral relationships
def f_team(soc_network,n,team,scores,wplus,wminus,wneutr):
    #Raise errors
    if not (0 <= wplus <= 1) or not (0 <= wneutr <=1) or not (0 <= wminus <=1):
        raise ValueError('The weights (wplus, wneutral and wminus) have to be between 0 and 1.')
    weights = soc_network[:,2]
    if any((weight > 1) or (weight < -1) for weight in weights):
        raise ValueError('All the weights in edge list have to have value 1 or -1 or in special cases 0')
    if any(float(weight).is_integer() == False for weight in weights):
        raise ValueError('All the weights in edge list have to be integers')

    A = np.zeros((n,n), dtype=np.int)
    for rows in soc_network:
        A[rows[0,0],rows[0,1]]=rows[0,2]
    A = np.matrix(A)
    A = A[np.ix_(team,team)]
    i=0
    team_f=0
    for rows in A:
        plus = np.where(rows==1)[0].size
        minus = np.where(rows==-1)[0].size
        neutr = np.where(rows==0)[0].size
        node_f = scores[i]*(plus*wplus - minus*wminus + neutr*wneutr)/(team.size-1)
        i += 1
        team_f += node_f
    return (team_f)




#f_total calculates the fitness function of the solution x. It is a sum of all the fitness functions of all the teams in the solution
#INPUTS:
#x - solution - an array of size n. If (for example) in_team=3: first 3 elements are indices of students in 1st team, second 3 elements are students in second team, etc.
#n - a number of students in class
#in_team - number of students in each team
#scores - the scores of the class
#wplus - weighting the positive relationships
#wminus - weighting the negative relationships
#wneutr - weighting the neutral relationships
def f_total(x,soc_network,n,in_team,scores,wplus=1,wminus=0,wneutr=0.5):
    #Raise errors:
    if float(n/in_team).is_integer()==False:
        raise ValueError('The total number of students has to be divisible by the number of students in team')
    if len(scores)!=int(n):
        raise ValueError('The length of scores list has to be equal with the number of students in the class')

    num_teams=int(n/in_team)
    x = np.array(x)
    scores = np.array(scores)
    p = np.split(x, num_teams)
    f = 0
    for team in p:
        f += f_team(soc_network,n,team,scores,wplus,wminus,wneutr)
        #print(team,f)
    return(f,)

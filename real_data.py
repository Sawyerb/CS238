"""
Calculating distributions for:
Money - Votes
Votes - Polls
"""
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import warnings

def get_cands_votes(): 
    '''
    From FEC.
    Note: tere are some rows where the same candidate repeats
    with different party.
    '''
    cands = {}
    data = './data/2014_house_election_results.csv'
    with open(data, 'rU') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            if row['FEC ID#'] in cands and row['GENERAL %'] != '':
                cands[row['FEC ID#']]['vote %'] += float(row['GENERAL %'].replace('%', ''))
                if cands[row['FEC ID#']]['party'] not in ['R', 'D'] and row['PARTY'] in ['R', 'D']: 
                    cands[row['FEC ID#']]['party'] = row['PARTY']
            elif row['FEC ID#'] != 'n/a' and row['FEC ID#'] != '' and row['GENERAL %'] != '':
                person = {}
                person['name'] = row['CANDIDATE NAME']
                person['state'] = row['STATE ABBREVIATION']
                person['district'] = row['D'].replace(' - FULL TERM', '').replace(' - UNEXPIRED TERM', '')
                party = row['PARTY']
                if '/' in party: 
                    parties = party.split('/')
                    if 'D' in parties: 
                        party = 'D'
                    elif 'R' in parties: 
                        party = 'R'
                    else: 
                        party = row['PARTY']
                if party == 'DFL': party = 'D' # MN's dem party
                person['party'] = party.strip()
                person['vote %'] = float(row['GENERAL %'].replace('%', ''))
                cands[row['FEC ID#'].strip()] = person
    return cands

def get_money(): 
    '''
    From FEC.
    ./data/cn 2.txt is less useful than expected because for H 2014 there are often multiple Dem and Rep.
    '''
    money = defaultdict(float)
    cand_campaign_finance = './data/webl14.txt'
    with open(cand_campaign_finance, 'r') as infile:
        for line in infile:
            contents = line.strip().split('|')
            money[contents[0].strip()] = float(contents[17].strip())
    return money

def get_polls(): 
    dem = {}
    rep = {}
    data = './data/2014_house_election_polls.csv'
    with open(data, 'rU') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            dem[row['CD']] = float(row['Dem'].replace('%', ''))
            rep[row['CD']] = float(row['Rep'].replace('%', ''))
    return dem, rep

def organize_data(): 
    cands = get_cands_votes()
    money = get_money()
    dem, rep = get_polls()
    total_election_money = defaultdict(float)
    for cand in cands:
        total_election_money[cands[cand]['state'] + cands[cand]['district']] += money[cand]
    file = open('./data/2014_house_election_clean.txt', 'w')
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['state', 'district', 'id', 'name', 'votes %', 'money %', 'poll %'])
    for cand in cands:
        if money[cand] != 0.0 and cands[cand]['state'] not in ['DC', 'AS', 'VI', 'GU']:
            money_percent = money[cand] * 100 / total_election_money[cands[cand]['state'] + cands[cand]['district']]
            district = cands[cand]['district']
            if district == '00':
                district = '01'
            if cands[cand]['party'] == 'D':
                writer.writerow([cands[cand]['state'], cands[cand]['district'], cand, cands[cand]['name'], 
                    cands[cand]['vote %'], money_percent, dem[cands[cand]['state']+district]])
            if cands[cand]['party'] == 'R':
                writer.writerow([cands[cand]['state'], cands[cand]['district'], cand, cands[cand]['name'], 
                    cands[cand]['vote %'], money_percent, rep[cands[cand]['state']+district]])
    file.close()

def find_best_fit(data): 
    '''
    Based on "Distribution Fitting with Sum of Square Error (SSE)"
    https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
    '''
    print "Finding best fit..."
    y, x = np.histogram(data, bins=200, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    all_dists = [
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    ]
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf
    for d in all_dists:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                params = d.fit(data)
                pdf = d.pdf(x, loc=params[-2], scale=params[-1], *params[:-2])
                sse = np.sum(np.power(y - pdf, 2.0)) # sum of square error
                if best_sse > sse > 0:
                    best_distribution = d
                    best_params = params
                    best_sse = sse
        except Exception:
            pass
    print "Got best fit..."
    return (best_distribution.name, best_params)

def visualize_relationships(): 
    votes = []
    money = []
    poll = []
    with open('./data/2014_house_election_clean.txt', 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            votes.append(float(row['votes %']))
            money.append(float(row['money %']))
            poll.append(float(row['poll %']))
    # % Money - % Votes
    # plot of values against values
    plt.scatter(money, votes)
    plt.xlabel('money %')
    plt.ylabel('votes %')
    plt.savefig('./data/money_vs_votes.png')
    plt.close()
    # plot of 1% money against % votes
    plt.title('vote percent per money percent')
    x = np.array(votes) / np.array(money)
    x = x[~np.isinf(x)]
    x = x[x < 10] # remove outliers
    np.save('./data/vote_per_money.npy', x)
    best_fit_name, best_fit_params = find_best_fit(x)
    print "Best fit for vote_per_money:", best_fit_name, best_fit_params
    best_dist = getattr(st, best_fit_name)
    plt.plot(sorted(x), best_dist.pdf(sorted(x), loc=best_fit_params[-2], scale=best_fit_params[-1], *best_fit_params[:-2]))
    #a, b, loc, scale = st.beta.fit(x)
    #plt.plot(sorted(x), st.beta.pdf(sorted(x), a, b, loc, scale), label='beta')
    #nparam_density = st.kde.gaussian_kde(x)
    #plt.plot(sorted(x), nparam_density(sorted(x)), label='gaussian')
    plt.hist(x, density=True, bins=200)
    plt.legend()
    plt.savefig('./data/vote_per_money.png')
    plt.close()
    # % Votes - % Polls
    # plot of values against values
    plt.scatter(votes, poll)
    plt.xlabel('votes %')
    plt.ylabel('poll %')
    plt.savefig('./data/votes_vs_polls.png')
    plt.close()
    # plot of 1% of poll against % votes
    plt.title('poll percent per vote percent')
    x = np.array(poll) / np.array(votes)
    x = x[~np.isinf(x)]
    x = x[x < 10] # remove outliers
    np.save('./data/poll_per_vote.npy', x)
    best_fit_name, best_fit_params = find_best_fit(x)
    print "Best fit for poll per vote:", best_fit_name, best_fit_params
    best_dist = getattr(st, best_fit_name)
    plt.plot(sorted(x), best_dist.pdf(sorted(x), loc=best_fit_params[-2], scale=best_fit_params[-1], *best_fit_params[:-2]))
    #a, b, loc, scale = st.beta.fit(x)
    #plt.plot(sorted(x), st.beta.pdf(sorted(x), a, b, loc, scale), label='beta')
    #nparam_density = st.kde.gaussian_kde(x)
    #plt.plot(sorted(x), nparam_density(sorted(x)), label='gaussian')
    plt.hist(x, density=True, bins=100)
    plt.legend()
    plt.savefig('./data/poll_per_vote.png')
    plt.close()

def main(): 
    '''
    Output format:
    state | district | ID | name | votes | money | poll
    '''
    visualize_relationships()

if __name__ == '__main__':
    main()
import numpy as np
import scipy.stats
from scipy.special import comb
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
from compare_roc_auc import fast_delong

@wrap_non_picklable_objects    
def get_counts(seq, states):
    next_count = {a: 0 for a in states}
    cond_count = {a: {b: 0 for b in states} for a in states}
    num_tr = len(seq) - 1
    # Compute next and conditional counts
    for i in np.arange(1, len(seq)):
        for a in states:
            if seq[i - 1] == a:
                for b in states:
                    if seq[i] == b:
                        cond_count[a][b] += 1
                        next_count[b] += 1
                        break
                break
    cond_count_list = []
    for a in states:
        for b in states:
            cond_count_list.append(cond_count[a][b])

    next_count_list = []
    for a in states:
        next_count_list.append(next_count[a])
    return next_count_list, cond_count_list

@wrap_non_picklable_objects    
def compile_sequence_counts(seq_list, states):
    next_counts = []
    cond_counts = []    
    for seq in seq_list:
        count_res = get_counts(seq, states)
        next_counts.append(count_res[0])
        cond_counts.append(count_res[1])
    return np.array(next_counts), np.array(cond_counts)

@wrap_non_picklable_objects    
def get_L_star_vals(a, b, next_counts, cond_counts, use_mean_rates=True):
    num_states = next_counts.shape[1]
    # Column indices where next != a (i.e., transitions in T_{A_complement})
    a_comp_ind = (
        np.array([i for i in range(num_states) if i != a])
    )
    # Count transitions where prev == a and next != a
    a_comp_cond_sum = cond_counts[:, a_comp_ind + a*num_states].sum(axis=1)
    if use_mean_rates:      
        # Compute L_star using base rates averaged over the whole sample
        # of sequences; note that as opposed to the computation of
        # L_star below, we only exclude samples with P(b|a) == nan; that is,
        # we only exclude sequences with no transitions from a to another state
        sample_pos = np.flatnonzero(
            a_comp_cond_sum > 0
        )
        # Compute mean base rate of b restricted to transitions with next != a
        modified_mean_base_rate = np.mean(
            next_counts[sample_pos, b] /
            next_counts[sample_pos, :][:, a_comp_ind].sum(axis=1)
        )
        # Compute conditional rate of b restricted to transitions with next != a
        cond_rates = (
            cond_counts[sample_pos, a*num_states + b] /
            a_comp_cond_sum[sample_pos]
        )
        L_star_vals = (
            (cond_rates - modified_mean_base_rate)
            / (1 - modified_mean_base_rate)
        )
    else:
        # Compute L_star using base rates from each individual sequence

        # Column indices where next != a and next != b
        a_b_comp_ind = (
            np.array([i for i in range(num_states) if i != a and i != b])
        )
        # Count transitions where next != a or next != b
        a_b_comp_sum = next_counts[:, a_b_comp_ind].sum(axis=1)
        # Count transitions where next != a        
        a_comp_sum = next_counts[:, b] + a_b_comp_sum
        # Find samples where:
        #  (a) P(b|a) != nan
        #  (b) P(b) < 1
        sample_pos = np.flatnonzero(
            (a_comp_cond_sum > 0) & (a_b_comp_sum > 0)            
        )        
        # Compute base rates of b restricted to transitions with next != a
        modified_base = (
            next_counts[sample_pos, b] / a_comp_sum[sample_pos]
        )
        # Compute conditional rate of b restricted to transitions with next != a
        cond_rates = (
            cond_counts[sample_pos, a*num_states + b] /
            a_comp_cond_sum[sample_pos]
        )       
        L_star_vals = (
            (cond_rates - modified_base)
            / (1 - modified_base)
        )
    return L_star_vals

@wrap_non_picklable_objects    
def sequences_to_y_X(seq_list, a, b):
    """ Function for turning a list of sequences into the appropriate
    format for GEE model
    Parameters
    ----------
    seq_list : list of lists
        Each entry in the list is a sequence (list) of transition states
        Example:
            [
                ['A', 'C', 'C', 'B', 'C'],
                ['B', 'C', 'A', 'C'],
                ['C', 'C', 'C', 'B', 'B', 'A']
            ]
    a : str/float
        Starting state
    b : str/float
        Ending state
    Returns
    -------
    y : 1d numpy array
        Array of dependent variables ("endogeneous" variables)    
    X : 2d numpy array
        Array of independent variables ("exogeneous" variables)
    seq_ind : 1d numpy array
        Array containing cluster/group labels for GEE model
    """
    y = []
    X = []
    seq_ind = []
    for i in range(len(seq_list)):
        curr_seq = seq_list[i]
        for j in range(len(curr_seq) - 1):
            seq_ind.append(i)                
            if curr_seq[j] == a:
                X.append([1, 1])
            else:
                X.append([1, 0])
            if curr_seq[j + 1] == b:
                y.append(1)
            else:
                y.append(0)

    return np.array(y), np.array(X), np.array(seq_ind)

@wrap_non_picklable_objects    
def generate_sequence(seq_length, state_dict, base_rates,
                                dependent_rates):
    # np.random.multinomial is faster than np.random.choice
    state_ind = np.arange(len(state_dict))
    inv_state_dict = {}
    new_rates = dependent_rates.copy()    
    for i in range(len(state_dict)):
        inv_state_dict[state_dict[i]] = i
        new_rates[i, :] += base_rates
        new_rates[i, :] /= new_rates[i, :].sum()
    
    seq = [state_dict[np.argmax(np.random.multinomial(1, base_rates))]] 

    for i in range(seq_length - 1):        
        temp_rates = new_rates[inv_state_dict[seq[-1]], :]
        seq.append(state_dict[np.argmax(np.random.multinomial(
            1, temp_rates))])
    return seq

def run_sequence_sims(rate=0.0, seq_length=20, num_trials=50,
                      num_runs=10000, verbose=5, n_jobs=1):

    states = ['A', 'B', 'C', 'D', 'E']
    base_rates = np.ones(len(states)) / len(states)
    num_states = len(states)
    state_dict = {}
    for i in range(num_states):
        state_dict[i] = states[i]
        
    dep_rates = np.zeros((num_states, num_states))
    # When rate > 0 we have four false null hypotheses
    dep_rates[0, 1] = rate
    dep_rates[0, 3] = -1*rate
    dep_rates[2, 1] = -1*rate
    dep_rates[2, 3] = rate

    out = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(sequence_sim)(
        seq_length,
        num_trials,
        state_dict,
        base_rates,
        states,
        dep_rates,
        num_states)
        for x in range(num_runs))
    return out

@wrap_non_picklable_objects    
def sequence_sim(seq_length, num_trials, state_dict, base_rates, states,
               dep_rates, num_states):

    res_list_gee = []
    val_list_gee = []
    beta_list = []
    val_list = []
    res_list = []
    L_list = []  
    
    seq_list = []
    for j in range(num_trials):
        seq_list.append(generate_sequence(seq_length, state_dict,
                                          base_rates, dep_rates))
    next_counts, cond_counts = compile_sequence_counts(seq_list, states)
    for m in range(num_states):
        for n in range(num_states):
            if m != n:
                # Compute L_star values
                res = get_L_star_vals(m, n, next_counts, cond_counts,
                                      use_mean_rates=False)
                t_res = scipy.stats.ttest_1samp(res, 0)

                res_list.append(t_res[1])
                val_list.append(t_res[0])
                L_list.append(np.mean(res))
            # Run marginal model 
            y, X, groups = sequences_to_y_X(seq_list, states[m], states[n])    
            md = sm.GEE(
                y, X, groups,
                cov_struct=sm.cov_struct.Exchangeable(),
                family=sm.families.Binomial()
            )                
            fit_res = md.fit(maxiter=60)

            res_list_gee.append(fit_res.pvalues[1])
            val_list_gee.append(fit_res.tvalues[1])
            beta_list.append(fit_res.params[1])
            
    return (
        # GEE p-values, test statistics, and beta coefficients        
        np.array(res_list_gee), np.array(val_list_gee), np.array(beta_list),
        # L_star p-values, test statistics, and computed values        
        np.array(res_list), np.array(val_list), np.array(L_list)
    )

def analyze_sequence_results(sim_data, dependence=False, L_star=False,
                               thresh_list = [0.05, 0.1, 0.15]):
    '''
    Analyze the output from run_sequence_sims.  Returns the estimated FDR 
    values from the BH and BY procedures, along with error bounds for the 
    99% confidence intervals.
    '''
    res_array = np.zeros((len(thresh_list), 4))    
    res_str = ''
       
    if L_star:
        res_ind = 3
        num_pairs = 5*4
        null_ind = np.arange(num_pairs)
        if dependence:
            false_null = np.array([0, 2, 9, 10])
            null_ind = np.setdiff1d(null_ind, false_null)            
    else:
        res_ind = 0
        num_pairs = 5*5
        null_ind = np.arange(num_pairs)
        if dependence:
            false_null = np.array([1, 3, 11, 13])
            null_ind = np.setdiff1d(null_ind, false_null)                    
    

    for thresh_ind in range(len(thresh_list)):
        thresh = thresh_list[thresh_ind]
        Q_array = np.zeros((len(sim_data), 2))        
        for i in range(len(sim_data)):        
            curr_res = sm.stats.multipletests(sim_data[i][res_ind],
                                              method='fdr_bh',
                                              alpha=thresh)        
            if len(np.flatnonzero(curr_res[0])) > 0:
                Q_array[i, 0] = (
                    len(np.flatnonzero(curr_res[0][null_ind]))/
                    len(np.flatnonzero(curr_res[0]))
                )
            curr_res = sm.stats.multipletests(sim_data[i][res_ind],
                                              method='fdr_by',
                                              alpha=thresh)
            if len(np.flatnonzero(curr_res[0])) > 0:
                Q_array[i, 1] = (
                    len(np.flatnonzero(curr_res[0][null_ind]))/
                    len(np.flatnonzero(curr_res[0]))
                )
        # Estimated FDR for BH (fdr[0]) and BY (fdr[1])
        fdr = np.mean(Q_array, axis=0)
        se = scipy.stats.sem(Q_array, axis=0)
        # 99% error for BH               
        err0 = se[0]*scipy.stats.t.ppf((1+0.99)/2, 
                                     Q_array.shape[0] - 1)
        # 99% error for BY                
        err1 = se[1]*scipy.stats.t.ppf((1+0.99)/2, 
                                     Q_array.shape[0] - 1)
        res_str += '{}: {:.3f} +/- {:.3f},  {:.3f} +/- {:.3f}\n'.format(
            thresh, fdr[0], err0, fdr[1], err1
        )        
        res_array[thresh_ind, :] = np.array([fdr[0], err0, fdr[1], err1])
    print(res_str)
    return res_array

@wrap_non_picklable_objects    
def classifier_sim(test_set_probs, test_size, num_models, classifier_sd,
                          inv_probs, num_cols):
    # Hack to get scipy.stats.norm to properly randomize
    # Without this, and when using n_jobs >= 2, the same numbers are generated
    # each simulation run    
    scipy.stats.norm.random_state.set_state(np.random.get_state())
    
    accuracy_array = np.zeros((1, num_models))
    auc_array = np.zeros((1, num_models))    
    res_array = np.zeros((1, num_cols))
    val_array = np.zeros((1, num_cols))
    res_array_auc = np.zeros((1, num_cols))
    val_array_auc = np.zeros((1, num_cols))     
    # generate answers
    y = np.random.binomial(1, test_set_probs)
    pos_ind = np.flatnonzero(y == 1)
    neg_ind = np.flatnonzero(y == 0)
    model_probs = np.zeros((test_size, num_models))
    for i in range(num_models):
        noise = scipy.stats.norm.rvs(size = test_size, scale = classifier_sd[i])
        
        model_probs[:, i] = scipy.stats.norm.cdf(inv_probs + noise)
    ind = 0
    model_preds = np.round(model_probs)
    acc_res = np.zeros((test_size, num_models))
    acc_res[model_preds == np.tile(np.array([y]).T, num_models)] = 1
    accuracy_array[0, :] = acc_res.mean(axis=0)
    for i in range(num_models - 1):
        for j in range(i+1, num_models):
            # Run McNemar's test
            table = np.zeros((2, 2))
            for m in range(2):
                for n in range(2):
                    table[m, n] = len(np.flatnonzero(
                                (acc_res[:, i] == m) & (acc_res[:, j] == n)))
            m_res = mcnemar(table)
            res_array[0, ind] = m_res.pvalue.copy()
            val_array[0, ind] = m_res.statistic.copy()
            # Run DeLong's test
            theta, p_val, z_stat = fast_delong(
                np.r_[model_probs[pos_ind][:, [i, j]],
                      model_probs[neg_ind][:, [i, j]]],
                len(pos_ind))
            val_array_auc[0, ind] = z_stat
            res_array_auc[0, ind] = p_val
            ind += 1
        auc_array[0, i] = theta[0]
        if i == num_models - 2:
            auc_array[0, -1] = theta[1]
   
    return (
        # accuracy p-values, test statistics, and computed values
        res_array, val_array,  accuracy_array,
        # AUROC p-values, test statistics, and computed values            
        res_array_auc, val_array_auc, auc_array
    )
                   
def run_classifier_sims(classifier_sd=[0.1, 0.1, 0.1, 0.5, 1, 2],
                        test_size=500, num_runs=10000, verbose=5, n_jobs=1):

    test_set_probs = np.random.uniform(0.01, 0.99, size=test_size)
    inv_probs = scipy.stats.norm.ppf(test_set_probs)
    num_models = len(classifier_sd)
    
    accuracy_array = np.zeros((num_runs, num_models))
    num_cols = int(comb(num_models, 2))
    out = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(classifier_sim)(
        test_set_probs,
        test_size,
        num_models,
        classifier_sd,
        inv_probs,
        num_cols)
        for x in range(num_runs))    
    return out

def analyze_classifier_results(sim_data, classifier_sd, AUROC=False,
                               thresh_list = [0.05, 0.1, 0.15]):
    '''
    Analyze the output from run_classifier_sims.  Returns the estimated FDR 
    values from the BH and BY procedures, along with error bounds for the 
    99% confidence intervals.
    '''
    res_array = np.zeros((len(thresh_list), 4))
    res_str = ''    
    num_models = len(classifier_sd)

    res_ind = 0
    if AUROC:
        res_ind = 3

    null_ind = []
    ind = 0
    for i in range(num_models):
        for j in range(i + 1, num_models):
            if classifier_sd[i] == classifier_sd[j]:
                null_ind.append(ind)
            ind += 1

    for thresh_ind in range(len(thresh_list)):
        thresh = thresh_list[thresh_ind]
        Q_array = np.zeros((len(sim_data), 2))
        for i in range(len(sim_data)):        
            curr_res = sm.stats.multipletests(sim_data[i][res_ind][0],
                                              method='fdr_bh',                                      
                                              alpha=thresh)        
            if len(np.flatnonzero(curr_res[0])) > 0:
                Q_array[i, 0] = (
                    len(np.flatnonzero(curr_res[0][null_ind]))/
                    len(np.flatnonzero(curr_res[0]))
                )
            curr_res = sm.stats.multipletests(sim_data[i][res_ind][0],
                                              method='fdr_by',                                      
                                              alpha=thresh)
            if len(np.flatnonzero(curr_res[0])) > 0:
                Q_array[i, 1] = (
                    len(np.flatnonzero(curr_res[0][null_ind]))/
                    len(np.flatnonzero(curr_res[0]))
                )
        # Estimated FDR for BH (fdr[0]) and BY (fdr[1])
        fdr = np.mean(Q_array, axis=0)
        se = scipy.stats.sem(Q_array[:, :2])
        # 99% error for BH
        err0 = se[0]*scipy.stats.t.ppf((1 + 0.99)/2, 
                                       Q_array.shape[0] - 1)
        # 99% error for BY        
        err1 = se[1]*scipy.stats.t.ppf((1 + 0.99)/2, 
                                       Q_array.shape[0] - 1)
        res_str += '{}: {:.3f} +/- {:.3f},  {:.3f} +/- {:.3f}\n'.format(
            thresh, fdr[0], err0, fdr[1], err1
        )        
        res_array[thresh_ind, :] = np.array([fdr[0], err0, fdr[1], err1])    
    print(res_str)
    return res_array

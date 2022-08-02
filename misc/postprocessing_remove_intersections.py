from utils import parse_vtt_sub_file, seconds_to_string

import os 
import pickle
import numpy as np 
from tqdm import tqdm
from config.config import load_opts

opts = load_opts()

def postprocessing_remove_intersections(test_file, path_subtitles = '/scratch/shared/beegfs/albanie/shared-datasets/bobsl/public_dataset_release/subtitles/audio-aligned', 
                                                    path_probabilities = 'inference_output/probabilities/', 
                                                    path_postpro_subs = 'inference_output/postprocessing/', 
                                                    threshold = 0.4,
                                                    max_shift = 0,
                                                    width_shift = 1, 
                                                    stride = 4):
    # try:
    if 1:
        if os.path.exists(os.path.join(path_subtitles, test_file + '/signhd.vtt')):
            sub_ext = '/signhd.vtt'
        elif os.path.exists(os.path.join(path_subtitles, test_file + '.vtt')):
            sub_ext = '.vtt'
        else: 
            print('cannot find file ', os.path.join(path_subtitles, test_file))
        path_pred = os.path.join(path_subtitles, test_file + sub_ext)

        # load preds 
        probs = pickle.load(open(os.path.join(path_probabilities, test_file, 'out.pkl'), 'rb'))
        preds = probs['preds']
        preds = [np.repeat(p,stride) for p in preds]
        st_end = np.round(np.array(probs['wind_fr_to'])*25).astype(int)

        parsed_pred = parse_vtt_sub_file(path_pred)

        # take averages over the predictions 
        repeats = np.arange(-max_shift, max_shift+1, width_shift)
        # repeats = np.array([0])

        parsed_pred_averages = []
        preds_averages = []
        st_end_averages = []
        for i in range(0, len(parsed_pred), len(repeats)): 
            median_start = np.median(np.array([parsed_pred[i + j][1] for j in range(len(repeats))]))
            median_end = np.median(np.array([parsed_pred[i + j][2] for j in range(len(repeats))]))
            parsed_pred_averages.append([parsed_pred[i][0], median_start, median_end])

            preds_vec = np.zeros((len(repeats), len(preds[0])+2*max_shift))
            for j in range(len(repeats)): 
                preds_vec[j, (repeats[j]+max_shift):(repeats[j]+max_shift + len(preds[i+j]))] = preds[i+j]
                if j==0:
                    st_end_averages.append([st_end[i][0], st_end[i][1]+2*max_shift])

            preds_vec = np.median(preds_vec, axis=0)
            preds_averages.append(preds_vec)

        parsed_pred = parsed_pred_averages 
        preds = preds_averages
        st_end = st_end_averages

        median_time_sub = np.array([np.mean([sub[1], sub[2]]) for sub in parsed_pred])
        ## chronological order subtitles
        chrono_order = [x for _,x in sorted(zip(median_time_sub,np.arange(len(median_time_sub))))]

        # video to frame 
        max_time = int(round(np.max(np.array([sub[2] for sub in parsed_pred]))*25))+1000 # end time of last subtitle 

        cost_full = np.zeros((max_time,len(parsed_pred)))

        for j in range(len(parsed_pred)): # for each subtitle
            sub_ind = chrono_order[j] # j = reshuffled index of sub, sub_ind is the ID of sub
            if st_end[sub_ind][0]<0:
                cost_full[0:st_end[sub_ind][1], j] = preds[sub_ind][-1*st_end[sub_ind][0]:] # j 
            else:
                cost_full[st_end[sub_ind][0]:(st_end[sub_ind][0]+len(preds[sub_ind])), j] = preds[sub_ind] # j 

        binary_cost_matrix = np.array((cost_full>=threshold) + 0)

        non_zero_cols = np.where(np.max(binary_cost_matrix, axis=1)==1)[0]

        zero_rows = np.where(np.max(binary_cost_matrix, axis=0)==0)[0]
        for i in zero_rows: 
            start_frame_orig = int(round((parsed_pred[chrono_order[i]][1])*25))
            end_frame_orig = int(round((parsed_pred[chrono_order[i]][2])*25))
            binary_cost_matrix[start_frame_orig:end_frame_orig, i] = 1

        zero_rows = np.where(np.max(binary_cost_matrix, axis=0)==0)[0]

        active_frame_indices = non_zero_cols[::4]
        binary_cost_matrix = binary_cost_matrix[active_frame_indices, :]
        cost_matrix = cost_full[active_frame_indices, :]

        # ## reduce matrix to where there are conflicts 
        # print(' cost mat', binary_cost_matrix[0:10,0:10])
        # print('max cost mat ', np.max(binary_cost_matrix, axis=1), len(np.max(binary_cost_matrix, axis=1)))

        # row_conflicts = np.sum(binary_cost_matrix, axis=1)
        # row_conflicts = [i for i in range(len(row_conflicts)) if row_conflicts[i]>1]
        # cost_matrix = cost_matrix[row_conflicts,:]
        # active_frame_indices_noconflict = np.array([active_frame_indices[i] for i in row_conflicts])

        # # ### other axis
        # non_zero_cost_red = (cost_matrix>=0.5)+0 # reduced non zero cost
        # col_conflicts = np.sum(non_zero_cost_red, axis=0)
        # col_conflicts = [i for i in range(len(col_conflicts)) if col_conflicts[i]>1]
        # cost_matrix = cost_matrix[:,col_conflicts]
        # chrono_order_noconflict = [chrono_order[i] for i in col_conflicts]

        chrono_order_noconflict = chrono_order
        active_frame_indices_noconflict = active_frame_indices

        cost_matrix = 1 - cost_matrix

        DTW = np.ones((len(active_frame_indices_noconflict),len(chrono_order_noconflict)))*np.inf
        DTW[0,0] = 0
        dict_neighbours = ['0_0']
        neighbours = [[0,0]]

        for i in tqdm(range(1,len(active_frame_indices_noconflict))):
            for j in range(1,len(chrono_order_noconflict)):
                dict_neighbours.append(str(i)+'_' +str(j))
                cost = cost_matrix[i,j]
                DTW[i, j] = cost + np.min((DTW[i - 1, j - 1], DTW[i-1, j]))
                if np.argmin((DTW[i - 1, j - 1], DTW[i-1, j]))==0:
                    neighbours.append([i-1, j-1])
                elif np.argmin((DTW[i - 1, j - 1],DTW[i-1,j]))==1:
                    neighbours.append([i-1,j])
                else:
                    print('error')

        dc_neigh = dict(zip(dict_neighbours, neighbours))

        Y = np.zeros((len(active_frame_indices_noconflict),len(chrono_order_noconflict)))
        Y[-1,-1]=1
        Y[0,0]=1
        i=len(active_frame_indices_noconflict)-1
        j=len(chrono_order_noconflict)-1
        newi = 1
        newj = 1
        while (i!=0 and j!=0):
            newi, newj = dc_neigh[str(i) + '_' + str(j)]
            Y[newi, newj] = 1
            i = newi
            j = newj

        ## reinsert conflicts in Y 
        Y_orig = Y
        # Y_orig = binary_cost_matrix
        # print(Y_orig.shape)
        # for i in range(len(row_conflicts)): 
        #     for j in range(len(col_conflicts)): 
        #         Y_orig[row_conflicts[i], col_conflicts[j]] = Y[i,j]

        time_sub_index = {}

        for i in range(len(chrono_order)): 
            start_ind = np.min(np.where(np.array(Y_orig[:,i])==1)[0])
            end_ind = np.max(np.where(np.array(Y_orig[:,i])==1)[0])
            start_frame = active_frame_indices_noconflict[start_ind]-2
            end_frame = active_frame_indices_noconflict[end_ind]+2
            
            #print(parsed_pred[chrono_order_noconflict[i]][1], start_frame/25)
            #print(parsed_pred[chrono_order_noconflict[i]][2], end_frame/25)

            parsed_pred[chrono_order_noconflict[i]][1] = start_frame/25
            parsed_pred[chrono_order_noconflict[i]][2] = end_frame/25

        out_file = os.path.join(path_postpro_subs, test_file+sub_ext)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        fw = open(out_file, 'w')
        fw.write('WEBVTT\n\n')

        for sub in parsed_pred: 
            pred_str_fr = seconds_to_string(sub[1])
            pred_str_to = seconds_to_string(sub[2])
            pred_str_fr_to = f'{pred_str_fr} --> {pred_str_to}'

            fw.write(f'{pred_str_fr_to}\n')
            fw.write(f'{sub[0]}\n\n')

        fw.close()

        ### assert same len as orig subs
        assert len(parsed_pred) == len(probs['preds'])
    else:
    # except:
        print('error for ', test_file)



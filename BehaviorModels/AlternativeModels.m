function [resp_est, prob_resp_est, resp] = AlternativeModels(df_dir, sub_id, mode)
%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Two alternative models that incorporated a single strategy: 
% A forward-dominant (FD) model, in which subjects were assumed to take a forward move if allowable, 
% or to take an info-max move if the forward move was not allowed, and an efficient-exploration (EE) model, 
% in which subjects were assumed to explore the maze most efficiently (i.e., preferentially moving to 
% previously unexplored grid spaces, regardless of the state estimate confidence level).
% The FD and EE models corresponded to aspects of the SS model in which the confidence level was always low
% and always high, respectively.
%
% input: df_dir: the path of the directory which contains the behavioral data files.
%        sub_id: the subject index. 1~26 : included behavioral/imaging/decoding analyses.
%                                   27   : included behavioral analyses.
%                                   28~33: excluded from the analyses.
%        mode: 'EE'(efficient-exploration strategy) or 'FD'(forward-dominant strategy)
% output: resp_est: the estimation of subjects' moving directions.
%         prob_resp_est: the probability of subjects' actual actions.
%         resp: the actual moving directions (the last step of each game and steps in which subjects made wrong 
%               action [selected impassable direction] was excluded).
%
%%%%%%%%%%%%%%%%%%%%%%%%%
%% make scene-state table
load([df_dir '/MAZE25.mat']);
dir_act=[3,4,2,1; 1,3,4,2; 4,2,1,3; 2,1,3,4];
maze=MAZE_BASE.*PATH{1};
scene=zeros(25,4);
for s=1:25
    for d=1:4
        doors=maze(s,dir_act(d,:))==0;
        scene(s,d)=sum([4,2,1].*~doors(1:3));
    end
end

%% make adjacent matrix
links = zeros(25,25+1);
for s=1:25
    links(s,MAZE_BASE(s,:)+1) = PATH{1}(s,:);
end
links = links(:,2:end);
Path_graph = graph(links);

%% load behavioral data & model parameters
dfiles = dir([df_dir '/Behavior/sub' num2str(sub_id) '_ses*.mat']);
Nses = length(dfiles);

state = [];
dirt = [];
resp = [];
autoresp = [];
scene_obs = [];
game = []; maxgame = 0;
predtrial = [];
scene_pred = [];

for ses = 1:Nses
    data = load(fullfile(dfiles(ses).folder,dfiles(ses).name));
    state = [state data.state(~data.select_impassable)];
    dirt = [dirt data.direction(~data.select_impassable)];
    resp = [resp data.move_resp(~data.select_impassable)];
    autoresp = [autoresp data.automove_resp(~data.select_impassable)];
    
    doors = ~data.doors(~data.select_impassable,1:3)'; %% doors: close=1, open=0
    scene_obs = [scene_obs sum(doors.*repmat([4 2 1]',1,size(doors,2)),1)];
    
    game = [game data.game(~data.select_impassable)+maxgame];
    maxgame = maxgame + max(data.game);
    
    predtrial = [predtrial ~isnan(data.scene_choice(~data.select_impassable))];
    scene_pred = [scene_pred data.scene_choice(~data.select_impassable)];
end
resp(isnan(resp)) = autoresp(isnan(resp));

load([df_dir '/model_parameters_est.mat']);
if strcmp(mode, 'FD')
    params = FD(sub_id,:);
elseif strcmp(mode, 'EE')
    params = EE(sub_id,:);
end

%% HMM

resp_est = [];
prob_resp_est = [];

if strcmp(mode, 'FD')
    alpha = params(1);
elseif strcmp(mode, 'EE')
    beta = params(1);
end
Nbt = params(2);

Ngame = max(game);
for g = 1:Ngame
    tmp_scene_obs = scene_obs(game==g);
    tmp_resp = resp(game==g);
    tmp_autoresp = autoresp(game==g);
    tmp_scene_pred = scene_pred(game==g);
    
    tmp_resp_est = zeros(1,length(tmp_resp));
    tmp_prob_resp_est = zeros(1,length(tmp_resp));
    
    Nstep = length(tmp_resp)-1;
    for t = 1:Nstep
        if t == 1
            [s_cand,d_cand] = find(scene==tmp_scene_obs(1));
            s_cand = s_cand';
            d_cand = d_cand';
            state_cand  = (s_cand-1)*4 + d_cand;
            exp_state = num2cell(s_cand');
            exp_stateidx = num2cell(state_cand');
            
            P_mt = ones(size(s_cand))/length(s_cand);
            mt_seq = ones(size(s_cand));
            
            prob_state_conf = [ones(size(s_cand));zeros(size(s_cand))]';
            bt_mode = ones(size(s_cand));
        end
        
        % action probability
        P_act_all = zeros(length(s_cand),3);
        for i = 1:length(s_cand)
            if strcmp(mode,'FD')
                [P_act_all(i,:),~] = action_selection(s_cand(i), d_cand(i), exp_state{i}, tmp_scene_obs(t), maze, Path_graph, mode, alpha, 0);
            elseif strcmp(mode,'EE')
                [P_act_all(i,:),~] = action_selection(s_cand(i), d_cand(i), exp_state{i}, tmp_scene_obs(t), maze, Path_graph, mode, 0, beta);
            end
        end
        P_act_all = repmat(P_mt',1,3).*P_act_all;
        P_act = sum(P_act_all,1);
        max_P_act = max(P_act);
        if ~isnan(tmp_autoresp(t)) % no response within allotted time
            tmp_resp_est(t) = NaN;
            tmp_prob_resp_est(t) = NaN;
        else
            if ismember(tmp_resp(t),find(P_act==max_P_act))
                tmp_resp_est(t) = tmp_resp(t);
                tmp_prob_resp_est(t) = max_P_act;
            else
                [~,tmp_resp_est(t)] = max(max_P_act);
                tmp_prob_resp_est(t) = P_act(tmp_resp(t));
            end
        end
        
        % next states
        ns_cand = zeros(size(s_cand));
        nd_cand = zeros(size(s_cand));
        uc_cand = zeros(size(s_cand));
        nbt_mode = zeros(size(s_cand));
        for i = 1:length(s_cand)
            act = dir_act(d_cand(i),tmp_resp(t));
            ns_cand(i) = maze(s_cand(i),act);
            nd_cand(i) = 5-act;
            uc_cand(i) = scene(ns_cand(i),nd_cand(i));
            if tmp_scene_obs(t+1) == uc_cand(i) %% update_mode
                if ismember(ns_cand(i),exp_state{i})
                    if t < Nstep
                        nact = dir_act(nd_cand(i),tmp_resp(t+1));
                        n_ns = maze(ns_cand(i),nact);
                        if ismember(n_ns, exp_state{i}) && ~ismember(tmp_scene_obs(t+1),[1 2 4])
                            nbt_mode(i) = 1;
                        else
                            nbt_mode(i) = 0;
                            exp_state{i} = [exp_state{i} ns_cand(i)];
                            exp_stateidx{i} = [exp_stateidx{i} (ns_cand(i)-1)*4+nd_cand(i)];
                        end
                    else
                        nbt_mode(i) = 1;
                    end
                else
                    nbt_mode(i) = 0;
                    exp_state{i} = [exp_state{i} ns_cand(i)];
                    exp_stateidx{i} = [exp_stateidx{i} (ns_cand(i)-1)*4+nd_cand(i)];
                end
            else %% backtrack_mode
                nbt_mode(i) = 1;
            end
        end
        state_cand_ud = state_cand(nbt_mode==0);        
        
        %P_act_ud_low = P_act_low(nbt_mode==0, tmp_resp(t))';
        %P_act_ud_high = P_act_high(nbt_mode==0, tmp_resp(t))';
        P_act_ud = P_act_all(nbt_mode==0);%%%%
        
        P_mt_ud = P_mt;
        P_mt_bt = P_mt;
        mt_seq_prev = mt_seq;
        
        for midx = unique(mt_seq)
            P_mt_ud(mt_seq==midx & nbt_mode==0) = sum(P_mt(mt_seq==midx & nbt_mode==0));
            P_mt_bt(mt_seq==midx & nbt_mode==1) = sum(P_mt(mt_seq==midx & nbt_mode==1));
            mt_seq(mt_seq==midx & nbt_mode==1) = mt_seq(mt_seq==midx & nbt_mode==1) + 2^(t-1);
        end
        P_mt_ud = P_mt_ud(nbt_mode==0);
        P_mt_bt = P_mt_bt(nbt_mode==1);
        
        % re-estimation for backtrack mode
        r_exp_state = {};
        r_exp_stateidx = {};
        rns_cand = [];
        rnd_cand = [];
        if sum(nbt_mode) > 0 % some backtrack-patterns
            if Nbt == 1
                [rs_cand,rd_cand] = find(scene==tmp_scene_obs(t));
                rs_cand = rs_cand';
                rd_cand = rd_cand';
                for j = 1:length(rs_cand)
                    ract = dir_act(rd_cand(j),tmp_resp(t));
                    rns = maze(rs_cand(j),ract);
                    rnd = 5-ract;
                    if scene(rns, rnd) == tmp_scene_obs(t+1)
                        rns_cand = [rns_cand rns];
                        rnd_cand = [rnd_cand rnd];
                        r_exp_state{j,1} = [rs_cand(j) rns];
                        r_exp_stateidx{j,1} = [(rs_cand(j)-1)*4+rd_cand(j) (rns-1)*4+rnd];
                    end
                end
                r_exp_state(cellfun('isempty',r_exp_state)) = [];
                r_exp_stateidx(cellfun('isempty',r_exp_stateidx)) = [];
                rn_state_cand = (rns_cand-1)*4 + rnd_cand;
            else
                if t-Nbt+1 > 0
                    btstart = t-Nbt+1;
                else
                    btstart = 1;
                end
                for tt = btstart:t
                    if tt == btstart
                        [rs_cand,rd_cand] = find(scene==tmp_scene_obs(tt));
                        rs_cand = rs_cand';
                        rd_cand = rd_cand';
                        r_exp_state = num2cell(rs_cand');
                        r_exp_stateidx = num2cell(((rs_cand-1)*4+rd_cand)');
                    end
                    rns_cand = [];
                    rnd_cand = [];
                    ruc_cand = [];
                    for k = 1:length(rs_cand)
                        ract = dir_act(rd_cand(k),tmp_resp(tt));
                        rns_cand = [rns_cand maze(rs_cand(k),ract)];
                        rnd_cand = [rnd_cand 5-ract];
                        ruc_cand = [ruc_cand scene(rns_cand(k),rnd_cand(k))];
                        r_exp_state{k} = [r_exp_state{k} maze(rs_cand(k),ract)];
                        r_exp_stateidx{k} = [r_exp_stateidx{k} (maze(rs_cand(k),ract)-1)*4+(5-ract)];
                    end
                    rs_cand = rns_cand(ruc_cand==tmp_scene_obs(tt+1));
                    rd_cand = rnd_cand(ruc_cand==tmp_scene_obs(tt+1));
                    r_exp_state = r_exp_state(ruc_cand==tmp_scene_obs(tt+1));
                    r_exp_stateidx = r_exp_stateidx(ruc_cand==tmp_scene_obs(tt+1));
                end
                rns_cand = rs_cand;
                rnd_cand = rd_cand;
                rn_state_cand = (rns_cand-1)*4 + rnd_cand;
            end
        end
        
        mt_seq_ud = mt_seq(nbt_mode==0);
        mt_seq_bt = mt_seq(nbt_mode==1);
        mt_seq_past_ud = mt_seq_prev(nbt_mode==0);
        
        if sum(nbt_mode==0) > 0
            P_ud_givenstate = zeros(1,sum(nbt_mode==0));
            uni_state_cand_ud = unique(state_cand_ud);
            for i = 1:length(uni_state_cand_ud)
                nbt_mode_givenstate = nbt_mode(state_cand==uni_state_cand_ud(i));
                P_ud_givenstate(state_cand_ud==uni_state_cand_ud(i)) = sum(nbt_mode_givenstate==0)/length(nbt_mode_givenstate);
            end
            %P_nst_ud_low = P_act_ud_low.*P_ud_givenstate;
            %P_nst_ud_high = P_act_ud_high.*P_ud_givenstate;
            P_nst_ud = P_act_ud.*P_ud_givenstate;%%%%
            for midx = unique(mt_seq_past_ud)
                %P_nst_ud_low(mt_seq_past_ud==midx) = P_nst_ud_low(mt_seq_past_ud==midx)/sum(P_nst_ud_low(mt_seq_past_ud==midx));
                %P_nst_ud_high(mt_seq_past_ud==midx) = P_nst_ud_high(mt_seq_past_ud==midx)/sum(P_nst_ud_high(mt_seq_past_ud==midx));
                P_nst_ud(mt_seq_past_ud==midx) = P_nst_ud(mt_seq_past_ud==midx)/sum(P_nst_ud(mt_seq_past_ud==midx));%%%%
            end
            P_nmt_ud = P_mt_ud;
        end
        
        if sum(nbt_mode) > 0 % BT mode exist
            if sum(nbt_mode==0) > 0 % UD mode exist
                P_nst_bt = ones(1, length(rns_cand)*length(unique(mt_seq_bt)))/length(rns_cand);
                [~,ia,~] = unique(mt_seq_bt,'stable');
                P_nmt_bt = reshape((P_mt_bt(ia)'*ones(1,length(rns_cand)))', 1, []);
                %P_nst_high = [P_nst_ud_high P_nst_bt];
                %P_nst_low = [P_nst_ud_low P_nst_bt];
                
                P_nmt = [P_nmt_ud P_nmt_bt];
                
                P_conf_ud = prob_state_conf(nbt_mode==0,:);
                P_conf_bt = prob_state_conf(nbt_mode==1,:);
                P_conf_bt_low = reshape((P_conf_bt(ia,1)*ones(1,length(rns_cand)))',[],1);
                P_conf_bt_high = reshape((P_conf_bt(ia,2)*ones(1,length(rns_cand)))',[],1);
                prob_state_conf_next = [P_conf_ud;[P_conf_bt_low, P_conf_bt_high]];
            else % UD mode not exist
                P_nst_bt = ones(1, length(rns_cand)*length(unique(mt_seq_bt)))/length(rns_cand);
                [~,ia,~] = unique(mt_seq_bt,'stable');
                P_nmt_bt = reshape((P_mt_bt(ia)'*ones(1,length(rns_cand)))', 1, []);
                %P_nst_high = P_nst_bt;
                %P_nst_low = P_nst_bt;
                
                P_nmt = P_nmt_bt;
                
                P_conf_bt = prob_state_conf(nbt_mode==1,:);
                P_conf_bt_low = reshape((P_conf_bt(ia,1)*ones(1,length(rns_cand)))',[],1);
                P_conf_bt_high = reshape((P_conf_bt(ia,2)*ones(1,length(rns_cand)))',[],1);
                prob_state_conf_next = [P_conf_bt_low, P_conf_bt_high];
            end
        else % BT mode not exist
            %P_nst_low = P_nst_ud_low;
            %P_nst_high = P_nst_ud_high;
            
            P_nmt = P_nmt_ud;
            
            prob_state_conf_next = prob_state_conf(nbt_mode==0,:);
        end
        
        if sum(nbt_mode) > 0 % BT mode exist
            mt_seq_bt = reshape((mt_seq_bt(ia)'*ones(1,length(rns_cand)))', 1, []);
            mt_seq = [mt_seq_ud, mt_seq_bt];
            P_mt = [P_mt(nbt_mode==0) reshape((P_mt_bt(ia)'*ones(1,length(rns_cand))/length(rns_cand))',1,[])];
            
            s_cand = [ns_cand(nbt_mode==0) repmat(rns_cand,1,length(ia))];
            d_cand = [nd_cand(nbt_mode==0) repmat(rnd_cand,1,length(ia))];
            exp_state = [exp_state(nbt_mode==0); repmat(r_exp_state,length(ia),1)];
            
            state_cand = (s_cand-1)*4 + d_cand;
            exp_stateidx = [exp_stateidx(nbt_mode==0); repmat(r_exp_stateidx,length(ia),1)];
            
            if t > 1
                mt_rep_bt = bt_mode(nbt_mode==1);
                mt_rep = [mt_rep(nbt_mode==0) reshape((mt_rep_bt(ia)'*ones(1,length(rns_cand)))',1,[])];
            end
            bt_mode = [zeros(1,sum(nbt_mode==0)) ones(1,length(rns_cand)*length(ia))];
            if t == 1
                mt_rep = zeros(1,length(bt_mode));
            end
        else
            mt_seq = mt_seq_ud;
            P_mt = P_mt(nbt_mode==0);
            %P_st = P_st(nbt_mode==0);
            
            s_cand = ns_cand(nbt_mode==0);
            d_cand = nd_cand(nbt_mode==0);
            exp_state = exp_state(nbt_mode==0);
            state_cand = (s_cand-1)*4 + d_cand;
            exp_stateidx = exp_stateidx(nbt_mode==0);
            
            if t > 1
                mt_rep = mt_rep(nbt_mode==0);
            end
            bt_mode = zeros(1,sum(nbt_mode==0));
            if t == 1
                mt_rep = zeros(1,length(bt_mode));
            end
        end
    end
    
    resp_est = [resp_est tmp_resp_est(1:end-1)];
    prob_resp_est = [prob_resp_est tmp_prob_resp_est(1:end-1)];
end

% state confidence in the prediction trials
predtrial = predtrial(~cast([diff(game)~=0 1],'logical'));

resp(~isnan(autoresp)) = NaN;
resp = resp(~cast([diff(game)~=0 1],'logical'));
end

function [p_move, optimal] = action_selection(s, d, exp_state, observation, maze, links, strategy, alpha, beta)
if strcmp(strategy, 'FD') % forward-dominant
    [p_move, optimal] = forward_dominant_strategy(observation, alpha);
elseif strcmp(strategy, 'EE') % efficient-exploration
    [p_move, optimal] = efficient_exploration_strategy(s, d, exp_state, maze, links, beta);
end
end

function [p_move, optimal] = forward_dominant_strategy(observation, alpha)
switch observation
    case 1
        p_move = [0,0,1];
        optimal = [0,0,1];
    case 2
        p_move = [0,1,0];
        optimal = [0,1,0];
    case 3
        p_move = [0,alpha,1-alpha];
        optimal = [0,1,-1];
    case 4
        p_move = [1,0,0];
        optimal = [1,0,0];
    case 5
        p_move = [1-alpha,0,alpha];
        optimal = [-1,0,1];
    case 6
        p_move = [1-alpha,alpha,0];
        optimal = [-1,1,0];
    case 7
        p_move = [(1-alpha)/2,alpha,(1-alpha)/2];
        optimal = [-1,1,-1];
end
end

function [p_move, optimal] = efficient_exploration_strategy(s, d, exp_state, maze, links, beta)
dir_act = [3,4,2,1; 1,3,4,2; 4,2,1,3; 2,1,3,4];
p_move = zeros(1,3);
optimal = zeros(1,3);

ns = maze(s, dir_act(d,1:3)); % next state if sbj move to L/F/R
optimal(ismember(ns, exp_state)) = -1; % possible but not toward-unexplored action
optimal(~ismember(ns, exp_state) & ns~=0) = 1; % possible and toward-unexplored action

if sum(optimal~=0) == 1 % only 1 possible way
    optimal(optimal~=0) = 1;
    p_move(optimal~=0) = 1;
    
else % possible way > 1
    if sum(optimal==1) == 0 % only non-toward-unexplored-state way
        [~, optresp] = mindist_unexpstate(s, d, exp_state, maze, links);
        optimal(optresp) = 1;
    end
    
    Nopt = sum(optimal==1);
    Npass = sum(optimal~=0);
    
    if Nopt == Npass
        p_move(optimal==1) = 1/sum(optimal==1);
    else
        p_move(optimal==1) = beta/sum(optimal==1);
        p_move(optimal==-1) = (1-beta)/sum(optimal==-1);
    end
end
end

function [mindist, optresp] = mindist_unexpstate(s, d, exp_state, maze, links)
dir_act = [3,4,2,1; 1,3,4,2; 4,2,1,3; 2,1,3,4];
Nstates = 1:25;
unexp_states = Nstates(~ismember(Nstates, exp_state));

dists = distances(links, s, unexp_states);
mindist = min(dists);
targetstate = unexp_states(dists==mindist);

optresp = [];

while 1
    for j = 1:length(targetstate)
    resp = 1:3;
        for button = resp
            act = dir_act(d, button);
            nstat = maze(s, act);
            if nstat ~= 0 % passable direction
                ndist = distances(links, nstat, targetstate(j));
                if ndist + 1 == mindist
                    optresp = [optresp button];
                end
            end
        end
    end
    
    if ~isempty(optresp)
        break;
    end
    
    dists(dists==mindist) = 100;
    mindist = min(dists);
    targetstate = unexp_states(dists==mindist);
end

optresp = unique(optresp);
end
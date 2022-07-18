clear all;
close all;
clc;
%%
[data,~]=text2bin('Hello World!');
numTrBlks = 1;              % Number of simulated transport blocks
simReps = [1];                % Repetitions to simulate


%% NPUSCH Configuration
ue = struct();                        % Initialize the UE structure
ue.NBULSubcarrierSpacing = '15kHz';   % 3.75kHz, 15kHz
ue.NNCellID = 10;                     % Narrowband cell identity

chs = struct();
chs.NPUSCHFormat = 'Data';        % NPUSCH payload type ('Data' or 'Control')
% The number of subcarriers used for NPUSCH 'NscRU' depends on the NPUSCH
% format and subcarrier spacing 'NBULSubcarrierSpacing' as shown in TS 36.211
% Table 10.1.2.3-1. There are 1,3,6 or 12 continuous subcarriers for NPUSCH
chs.NBULSubcarrierSet = 0:11;     % Range is 0-11 (15kHz); 0-47 (3.75kHz)
chs.NRUsc = length(chs.NBULSubcarrierSet);
% The symbol modulation depends on the NPUSCH format and NscRU as given by
% TS 36.211 Table 10.1.3.2-1
chs.Modulation = 'QPSK';
chs.CyclicShift = 0;        % Cyclic shift required when NRUsc = 3 or 6
chs.RNTI = 20;              % RNTI value
chs.NLayers = 1;            % Number of layers
chs.NRU = 1;                % Number of resource units
chs.SlotIdx = 0;            % The slot index
chs.NTurboDecIts = 5;       % Number of turbo decoder iterations
chs.CSI = 'On';             % Use channel CSI in PUSCH decoding

% RV offset signaled via DCI (See 36.213 16.5.1.2)
rvDCI = 0;
% Calculate the RVSeq used according to the RV offset
rvSeq = [2*mod(rvDCI+0,2)  2*mod(rvDCI+1,2)];

if strcmpi(chs.NPUSCHFormat,'Data')
    infoLen = length(data); %136;   % Transport block size for NPUSCH format 1
elseif strcmpi(chs.NPUSCHFormat,'Control')
    infoLen = 1;    % ACK/NACK bit for NPUSCH format 2
end

%% Propagation Channel Model Configuration
% The structure |channel| contains the channel model configuration
% parameters.
channel = struct;                    % Initialize channel config structure
channel.Seed = 6;                    % Channel seed
channel.NRxAnts = 1;                 % 2 receive antennas
channel.DelayProfile ='ETU';         % Delay profile
channel.DopplerFreq = 0;             % Doppler frequency in Hz
channel.MIMOCorrelation = 'Low';     % Multi-antenna correlation
channel.NTerms = 16;                 % Oscillators used in fading model
channel.ModelType = 'GMEDS';         % Rayleigh fading model type
channel.InitPhase = 'Random';        % Random initial phases
channel.NormalizePathGains = 'On';   % Normalize delay profile power
channel.NormalizeTxAnts = 'On';      % Normalize for transmit antennas

%% Channel Estimator Configuration
% In this example, the parameter |perfectChannelEstimator| controls channel
% estimator behavior. Valid values are |true| or |false|. When set to
% |true|, a perfect channel estimator is used. Otherwise a practical
% estimator is used, based on the values of the received NPUSCH DRS.

% Channel estimator behavior
perfectChannelEstimator = true;

%%
% For DRS signals in NPUSCH format 1, sequence-group hopping can be enabled
% or disabled by the higher layer cell-specific parameter
% |groupHoppingEnabled|. Sequence-group hopping for a particular UE can be
% disabled through the higher layer parameter |groupHoppingDisabled| as
% described in TS 36.211 Section 10.1.4.1.3 [ <#9 1> ]. In this example, we
% use the |SeqGroupHopping| parameter to enable or disable sequence-group
% hopping
chs.SeqGroupHopping = 'on'; % Enable/Disable Sequence-Group Hopping for UE
chs.SeqGroup = 0;           % Higher-layer parameter groupAssignmentNPUSCH

% Get number of time slots in a resource unit NULSlots according to
% TS 36.211 Table 10.1.2.3-1
if strcmpi(chs.NPUSCHFormat,'Data')
    if chs.NRUsc == 1
        NULSlots = 16;
    elseif any(chs.NRUsc == [3 6 12])
        NULSlots = 24/chs.NRUsc;
    else
        error('Invalid number of subcarriers. NRUsc must be one of 1,3,6,12');
    end
elseif strcmpi(chs.NPUSCHFormat,'Control')
    NULSlots = 4;
else
    error('Invalid NPUSCH Format (%s). NPUSCHFormat must be ''Data'' or ''Control''',chs.NPUSCHFormat);
end
chs.NULSlots = NULSlots;

%% Block Error Rate Simulation Loop
% To perform NB-IoT NPUSCH link level simulation and plot BLER results for
% a number of repetition levels, this example performs the following steps:
%
% For NPUSCH format 1 transmission for UL data transfer:
%
% * Generate a random stream of bits with the size of the desired transport
% block
% * Perform CRC encoding, turbo encoding and rate matching to create the
% NPUSCH bits
% * Interleave the bits per resource unit to apply a time-first mapping and
% create the NPUSCH codeword
%
% For NPUSCH format 2 used for signaling HARQ feedback for NPDSCH:
%
% * Perform bit repetition of the HARQ indicator to create the NPUSCH
% codeword
%
% Then for either NPUSCH format:
%
% * Perform scrambling, modulation, layer mapping and precoding on the
% codeword to form the complex NPUSCH symbols
% * Map the NPUSCH symbols and the corresponding DRS to the resource grid
% * Generate the time domain waveform by performing SC-FDMA modulation of
% the resource grid
% * Pass the waveform through a fading channel with AWGN
% * Recover the transmitted grid by performing synchronization, channel
% estimation and MMSE equalization
% * Extract the NPUSCH symbols
% * Recover the transport block by demodulating the symbols and channel
% decoding the resulting bit estimates
%
% Note that if practical channel estimation is configured
% (|perfectChannelEstimator = false|), practical timing estimation based on
% NPUSCH DRS correlation will also be performed. The timing offset is
% initialized to zero, intended to represent the initial synchronization
% after NPRACH reception. The timing estimate is then updated whenever the
% peak of the NPUSCH DRS correlation is sufficiently strong.
%
% After de-scrambling, the repetitive slots are soft-combined before rate
% recovery. The transport block error rate is calculated for each SNR
% point. The evaluation of the block error rate is based on the assumption
% that all the slots in a bundle are used to decode the transport block at
% the UE. A bundle is defined in the MAC layer (see 3GPP TS 36.321 5.4.2.1
% [ <#9 3> ]) as the |NPUSCH.NRU| $\times$ |NPUSCH.NULSlots| $\times$
% |NPUSCH.NRep| slots used to carry a transport block.

% Get the slot grid and number of slots per frame
emptySlotGrid = lteNBResourceGrid(ue); % Initialize empty slot grid
slotGridSize = size(emptySlotGrid);
NSlotsPerFrame = 20/(slotGridSize(1)/12);

tSlot = 10e-3/NSlotsPerFrame; % Slot duration
symbolsPerSlot = slotGridSize(2); % Number of symbols per slot

% Get a copy of the configuration variables ue, chs and channel to create
% independent simulation parfor loops
ueInit = ue;
chsInit = chs;
channelInit = channel;

%% Satellite Geometry and Parameter
beamwidth = 80.5*pi/180;             % Satellite antenna beamwidth
Gain = 2/(1-cos(beamwidth/2)) ;      % Ideal direction antenna gain
%% Fixed Parameters
fc = 921.6e6 ;                       % fc of the observer
Ro=earthRadius;             
kT = 290*physconst('Boltzmann');     % Noise Temperature 290 K
NF=10^(6/10);
h = 550e3;                           % Constellation Altitude
r  = Ro + h; 
env=[0.4 0 1 5 2];                   % Envirnomental factor of StG pathloss (Please refer to ITU-R for different Scenairo) the default [0.4 0 1 5 2] is for remote area
Fs = 10e6 ;                          % Sampling Freq
Fc = 921.5e6 ;                       % Carrier Freq of the observer
Servingcell = [];
bler=[];
WindowTime = 1 ;                     % This is in second
itr= 100;                            % iteration
BW=15e3+(chs.NRUsc-1)*15e3;          % Bandwidth of NB-IoT
N0=kT*BW*NF*1000;                    %Noise power
%% Parameter that we wants to optimise
Niot = ceil(510.1e9 * 2*0.061 / (24*3600*64));       % probablity of active user under the satellite N*NumofTxPerDay*packetlenght
Power   = [-10:2:2 2.1:0.1:3.9 4:2:30];
StartTxtime = linspace(1,2000,itr);              % Start Tx time (end time has to be less than the size of 3 column of Onduty)
%% pick up one device
% ChosenObserver = randi(Niot,1);      % one of the users in the script

%% Creating Tx
Obs_az = 2*pi*rand(1);
% Obs_el = pi/2; % These are with respect to ECI frame with 90 inc
Obs_el = [0:6:60].* pi/180; % These are with respect to ECI frame with 60 inc

%% creating traffic
for inccnt = 1:numel(Obs_el)
    for cnt = 1:itr
        clearvars DeltaX DeltaY DeltaZ Rk k axang Local_Sat el_iot az_iot x_iot y_iot z_iot sumIQ outputIQ DeltaX_Obs DeltaY_Obs DeltaZ_Obs
        Servingcell = [];
        load('SatelliteData\SatelliteDuty400_N_480_inc_600_WalkerS'); % this is the wallker constellation .mat you created using SatelliteOnDutyCalculation.m
                %% Create interferencer PPP
        el_iot = pi/2-acos(2*rand(Niot,1)-1); % These are with respect to ECI frame
        az_iot = 2*pi*rand(Niot,1);            %random point
        [x_iot,y_iot,z_iot] = sph2cart(az_iot,el_iot,1); % This is w.r.t. ECI
        %% Calculating Distance between iot and the sat
        
        [x_Obs,y_Obs,z_Obs] = sph2cart(Obs_az,Obs_el(inccnt),1); % This is w.r.t. ECI
        k = cross([x_Obs,y_Obs,z_Obs],...
            [0,0,1]);  % find the rotation axis
        axang = [k, (pi/2-Obs_el(inccnt))];        %rotate to
        Rk = axang2rotm(axang);
        Obs_Sat = Rk*[x_sat(:,ceil(StartTxtime(cnt)))';...
            y_sat(:,ceil(StartTxtime(cnt)))'; z_sat(:,ceil(StartTxtime(cnt)))'];
        DeltaX_Obs = Obs_Sat(1,:)-0;
        DeltaY_Obs = Obs_Sat(2,:)-0;
        DeltaZ_Obs = Obs_Sat(3,:)-1;      %doing another transformation
        [~,El_Obs,slantdist_Obs] = cart2sph(DeltaX_Obs,DeltaY_Obs,DeltaZ_Obs);
        DistanceStD_Obs = slantdist_Obs.*Ro ;
        [~,ServingCell] = min(DistanceStD_Obs) ;                  %find the serving satellite
        Servingslantdist=DistanceStD_Obs(ServingCell);
        ServingEl=El_Obs(ServingCell);
        varphi_Obs = acos ( ((r^2 + Ro^2 - Servingslantdist^2)/ (2 * r * Ro)) );        %find varphi of all iot to the serving satellite
        for ctr=1:Niot
            k = cross([x_iot(ctr),y_iot(ctr),z_iot(ctr)],...
                [0,0,1]);  % find the rotation axis
            axang = [k, (pi/2-el_iot(ctr))];        %rotate to
            Rk = axang2rotm(axang);
            Local_Sat = Rk*[x_sat(:,ceil(StartTxtime(cnt)))';...
                y_sat(:,ceil(StartTxtime(cnt)))'; z_sat(:,ceil(StartTxtime(cnt)))'];
            
            DeltaX = Local_Sat(1,:)-0;
            DeltaY = Local_Sat(2,:)-0;
            DeltaZ = Local_Sat(3,:)-1;      %doing another transformation
            
            [~,El(:,ctr),slantdist(:,ctr)] = cart2sph(DeltaX,DeltaY,DeltaZ);
        end
        
        DistanceStD = slantdist.*Ro ;
        
        %% find the interferencing cell
        varphi = acos ( ((r^2 + Ro^2 - DistanceStD(ServingCell,:).^2)/ (2 * r * Ro)) );        %find varphi of all iot to the serving satellite
        for repIdx = 1:numel(simReps)
            
            chsInit.NRep = simReps(repIdx); % Number of repetitions of the NPUSCH
            NSlotsPerBundle = chsInit.NRU*chsInit.NULSlots*chsInit.NRep; % Number of slots in a codeword bundle
            TotNSlots = numTrBlks*NSlotsPerBundle;   % Total number of simulated slots
            tottime=NSlotsPerBundle*tSlot;
            % Initialize BLER and throughput result
            maxThroughput = zeros(length(Power),1);
            simThroughput = zeros(length(Power),1);
            bler = zeros(1,numel(Power)); % Initialize BLER result
            
            for PowerIdx = 1:numel(Power)
                if ServingEl > 0
                    if beamwidth < 2*asin(alpha)
                        varphi_max = asin(sin(beamwidth/2)/alpha)-beamwidth/2 ;
                    else
                        varphi_max =  acos(alpha) ;
                    end
                    if varphi_Obs < varphi_max
                        interferencing_cell = varphi < varphi_max;
                        
                        %     randomstarttime = randi([-Samplelength,Samplelength],1,sum(interferencing_cell));
                        [FSPL,Eta] = StG_PathLoss(ServingEl*180/pi,Servingslantdist,Fc,[0.4 0 1 5 2]);
                        
                        
                        
                        
                        % parfor snrIdx = 1:numel(SNRdB)
                        % To enable the use of parallel computing for increased speed comment out
                        % the 'for' statement above and uncomment the 'parfor' statement below.
                        % This needs the Parallel Computing Toolbox (TM). If this is not installed
                        % 'parfor' will default to the normal 'for' statement.
                        
                        % Set the random number generator seed depending on the loop variable
                        % to ensure independent random streams
                        rng(PowerIdx,'combRecursive');
                        
                        ue = ueInit;    % Initialize ue configuration
                        chs = chsInit;  % Initialize chs configuration
                        channel = channelInit; % Initialize fading channel configuration
                        numBlkErrors = 0;  % Number of transport blocks with errors
                        estate = struct('SlotIdx',chs.SlotIdx);  % Initialize NPUSCH encoder state
                        dstate = estate;   % Initialize NPUSCH decoder state
                        offset = 0;        % Initialize overall frame timing offset
                        trblk = [];        % Initialize the transport block
                        npuschHest = [];   % Initialize channel estimate
                        noiseEst = [];     % Initialize noise estimate
                        
                        % Display the number of slots being generated
                        %             fprintf('\nGenerating %d slots corresponding to %d transport block(s) at %gdB Power\n',TotNSlots,numTrBlks,Power(PowerIdx));
                        
                        for slotIdx = 0+(0:TotNSlots-1)
                            % Calculate the frame number and slot number within the frame
                            ue.NFrame = fix(slotIdx/NSlotsPerFrame);
                            ue.NSlot = mod(slotIdx,NSlotsPerFrame);
                            % Create the slot grid
                            slotGrid = emptySlotGrid;
                            
                            if isempty(trblk)
                                
                                % Initialize transport channel decoder state
                                dstateULSCH = [];
                                
                                if strcmpi(chs.NPUSCHFormat,'Data')
                                    % UL-SCH encoding is performed for the two RV values used for
                                    % transmitting the codewords. The RV sequence used is determined
                                    % from the rvDCI value signaled in the DCI and alternates
                                    % between 0 and 2 as given in TS 36.213 Section 16.5.1.2
                                    
                                    % Define the transport block which will be encoded to create the
                                    % codewords for different RV
                                    trblk = data;
                                    % Determine the coded transport block size
                                    [~, info] = lteNPUSCHIndices(ue,chs);
                                    outblklen = info.G;
                                    % Create the codewords corresponding to the two RV values used
                                    % in the first and second block, this will be repeated till all
                                    % blocks are transmitted
                                    chs.RV = rvSeq(1); % RV for the first block
                                    cw = lteNULSCH(chs,outblklen,trblk); % CRC and Turbo coding is repeated
                                    chs.RV = rvSeq(2); % RV for the second block
                                    cw = [cw lteNULSCH(chs,outblklen,trblk)]; %#ok<AGROW> % CRC and Turbo coding is repeated
                                else
                                    trblk = randi([0 1],1); % 1 bit ACK
                                    % For ACK, the same codeword is transmitted every block as
                                    % defined in TS 36.212 Section 6.3.3
                                    cw = lteNULSCH(trblk);
                                end
                                blockIdx = 0; % First block to be transmitted
                            end
                            
                            % Copy SlotIdx for the SCFDMA modulator
                            chs.SlotIdx = estate.SlotIdx;
                            
                            % Set the RV used for the current transport block
                            chs.RV = rvSeq(mod(blockIdx,size(rvSeq,2))+1);
                            %% Mapping
                            
                            chs.NPUSCHPower = Power(PowerIdx);
                            chs.NPUSCHDRSPower = Power(PowerIdx)+4;
                            % NPUSCH encoding and mapping onto the slot grid
                            txsym = lteNPUSCH(ue,chs,cw(:,mod(blockIdx,size(cw,2))+1),estate);
                            slotGrid(lteNPUSCHIndices(ue,chs)) =  txsym*db2mag(chs.NPUSCHPower);
                            
                            % NPUSCH DRS and mapping on to the slot grid
                            [dmrs,estate] = lteNPUSCHDRS(ue,chs,estate);
                            slotGrid(lteNPUSCHDRSIndices(ue,chs)) = dmrs*db2mag(chs.NPUSCHDRSPower);
                            
                            % If a full block is transmitted, increment the clock counter so that
                            % the correct codeword can be selected
                            if estate.EndOfBlk
                                blockIdx = blockIdx + 1;
                            end
                            %% Modulation
                            % Perform SC-FDMA modulation to create the time domain waveform
                            [signalIQ,scfdmaInfo] = lteSCFDMAModulate(ue,chs,slotGrid);
                            
                            % Add 25 sample padding. This is to cover the range of delays
                            % expected from channel modeling (a combination of
                            % implementation delay and channel delay spread)
                            signalIQ =  [signalIQ; zeros(25, size(signalIQ,2))]; %#ok<AGROW>
                            
                            % Initialize channel time for each slot
                            channel.InitTime = slotIdx*tSlot;
                            
                            % Pass data through channel model
                            channel.SamplingRate = scfdmaInfo.SamplingRate;
                            Fs = scfdmaInfo.SamplingRate;
                            Nsamples = 5000;         %
                            randomstarttime = randi(Nsamples-1000,1,sum(interferencing_cell));
                            [signalIQ,fadingInfo] = lteFadingChannel(channel, signalIQ);
                            % Calculate noise gain
                            % SNR = 10^(SNRdB(PowerIdx)/20);
                            ObserverPacketLength = length(signalIQ) ;
                            %% adding FSPL Eta
                            outputIQ = signalIQ.*10^( -(FSPL + Eta) ./20);
                            outputIQ = [zeros(1,1000) outputIQ'] ;
                            sumIQ    = zeros(1, Nsamples) ;
                            % Normalize noise power to take account of sampling rate, which is
                            % a function of the IFFT size used in SC-FDMA modulation
                            %% adding interference
                            if sum(interferencing_cell) > 0
                                [FSPL_int,Eta_int] = StG_PathLoss(El(interferencing_cell)*180/pi,DistanceStD(interferencing_cell),Fc,[0.4 0 1 5 2]) ;
                                for n=1:sum(interferencing_cell)
                                    %% this part is calculating the interference signal
                                    % Perform SC-FDMA modulation to create the time domain waveform
                                    [InterIQ,scfdmaInfo] = lteSCFDMAModulate(ue,chs,slotGrid);
                                    % Pass data through channel model
                                    channel.SamplingRate = scfdmaInfo.SamplingRate;
                                    [InterIQ,fadingInfo] = lteFadingChannel(channel, InterIQ);
                                    % Add 25 sample padding. This is to cover the range of delays
                                    % expected from channel modeling (a combination of
                                    % implementation delay and channel delay spread)
                                    InterIQ =  [InterIQ; zeros(25, size(InterIQ,2))]; %#ok<AGROW>
                                    InterIQ = InterIQ.*10^( -(FSPL_int(n) + Eta_int(n)) ./ 20);
                                    InterIQ     = [zeros(1,randomstarttime(n)-1) InterIQ'] ;
                                    if length(InterIQ) < length(sumIQ)
                                        InterIQ = [InterIQ zeros(1,length(sumIQ)-length(InterIQ))] ;
                                    elseif length(InterIQ) > length(sumIQ)
                                        InterIQ = InterIQ(:,1:Nsamples) ;
                                    end
                                    sumIQ       = sumIQ + InterIQ ;
                                end
                            end
                            if length(outputIQ) < length(sumIQ)
                                outputIQ = [outputIQ zeros(1,length(sumIQ)-length(outputIQ))] ;
                            elseif length(outputIQ) > length(sumIQ)
                                error('Increase Time Window')
                            end
                            sumIQ       = sumIQ + outputIQ ;
                            %% Create additive white Gaussian noise
                            RxIQ =  sumIQ(:,1000:1000+ObserverPacketLength);
                            % Create additive white Gaussian noise
                            noise = wgn(1,length(RxIQ),N0/2,'linear')+1i * wgn(1,length(RxIQ),N0/2,'linear');
                            
                            RxIQ = RxIQ.*Gain.*(10*log10(Power(PowerIdx)));
                            % Add AWGN to the received time domain waveform
                            RxIQ = RxIQ + noise;
                            RxIQ=RxIQ';
                            %%%%%%%%%%%%%%%%%Receiver%%%%%%%%%%%%%%%%%%
                            if (perfectChannelEstimator)
                                offset = hPerfectTimingEstimate(fadingInfo);
                            else
                                [t,mag] = lteULFrameOffsetNPUSCH(ue, chs, RxIQ, dstate);
                                % The function hSkipWeakTimingOffset is used to update the
                                % receiver timing offset. If the correlation peak in 'mag'
                                % is weak, the current timing estimate 't' is ignored and
                                % the previous estimate 'offset' is used
                                offset = hSkipWeakTimingOffset(offset,t,mag);
                            end
                            
                            % Synchronize the received waveform
                            RxIQ = RxIQ(1+offset:end, :);
                            
                            % Perform SC-FDMA demodulation on the received data to recreate
                            % the resource grid, including padding in the event that
                            % practical synchronization results in an incomplete slot being
                            % demodulated
                            rxSlot = lteSCFDMADemodulate(ue,chs,RxIQ);
                            % Perfect channel estimation
                            ue.TotSlots = 1; % Channel estimate for 1 slot
                            estChannelGrid = lteULPerfectChannelEstimate(ue, chs, channel, offset);
                            noiseGrid = lteSCFDMADemodulate(ue,chs,noise(1+offset:end ,:));
                        
                            
                            % Get NPUSCH indices
                            npuschIndices = lteNPUSCHIndices(ue,chs);
                            % Get NPUSCH resource elements from the received slot
                            [rxNpuschSymbols, npuschHest] = lteExtractResources(npuschIndices, ...
                                rxSlot, estChannelGrid);
                            
                            % Perform channel estimate and noise estimate buffering in
                            % the case of practical channel estimation
                            % Decode NPUSCH
                         
                            [rxcw,dstate,symbols] = lteNPUSCHDecode(...
                                ue, chs, rxNpuschSymbols, npuschHest, N0,dstate);
                            
                            % Decode the transport block when all the slots in a block have
                            % been received
                            if dstate.EndOfBlk
                                % Soft-combining at transport channel decoder
                                [out, err, dstateULSCH] = lteNULSCHDecode(chs,infoLen,rxcw,dstateULSCH);
                            end
                            
                            % If all the slots in the bundle have been received, count the
                            % errors and reinitialize for the next bundle
                            if dstate.EndOfTx
                                if strcmpi(chs.NPUSCHFormat,'Control')
                                    err = ~isequal(out,trblk);
                                end
                                numBlkErrors = numBlkErrors + err;
                                % Re-initialize to enable the transmission of a new transport
                                % block
                                trblk = [];
                            end
                        end
                        % Calculate the block error rate
                        bler(PowerIdx) = numBlkErrors/numTrBlks;
                        %             fprintf('NPUSCH BLER = %.4f \n',bler(PowerIdx));
                        % Calculate the maximum and simulated throughput
                        maxThroughput(PowerIdx) = infoLen*numTrBlks; % Max possible throughput
                        simThroughput(PowerIdx) = infoLen*(numTrBlks-numBlkErrors);  % Simulated throughput
                        %             fprintf('NPUSCH Throughput(%%) = %.4f %%\n',simThroughput(PowerIdx)*100/maxThroughput(PowerIdx));
                        
                    else
                        bler(PowerIdx)= 1;
                    end
                else
                    bler(PowerIdx)= 1;
                end
            end            
        end
        Bler(cnt,:)=bler(:);
        bler=[];
    end
    BLER(inccnt,:)= mean(Bler,1);
    Bler=[];
end


clearvars DeltaX DeltaY DeltaZ Rk k axang Local_Sat el_iot az_iot x_iot y_iot z_iot interIQ sumIQ outputIQ NoiseIQ
save('PERvsTxpowerCase1_NBIoT');

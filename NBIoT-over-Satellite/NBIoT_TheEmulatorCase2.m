%% NB-IoT NPUSCH Block Error Rate Simulation
clear all;
close all;
clc;
%% Propagation Channel Model Configuration
% The structure |channel| contains the channel model configuration
% parameters.
channel = struct;                    % Initialize channel config structure
channel.Seed = 6;                    % Channel seed
channel.NRxAnts = 1;                 % 2 receive antennas
channel.DelayProfile ='off';         % Delay profile
channel.DopplerFreq = 0;             % Doppler frequency in Hz
channel.MIMOCorrelation = 'Low';     % Multi-antenna correlation
channel.NTerms = 32;                 % Oscillators used in fading model
channel.ModelType = 'GMEDS';         % Rayleigh fading model type
channel.InitPhase = 'Random';        % Random initial phases
channel.NormalizePathGains = 'On';   % Normalize delay profile power
channel.NormalizeTxAnts = 'On';      % Normalize for transmit antennas
perfectChannelEstimator = true;
%% Simulation Configuration
% The simulation length is 5 UL-SCH transport blocks for a number of SNR
% points |SNRdB| for different repetitions |simReps|. To produce meaningful
% throughput results, you should use a larger number of transport blocks
% (|numTrBlks|). |SNRdB| and |simReps| can be a specified as a scalar or a
% numeric array.
[data,~]=text2bin('Hello World!');
tbs = 1;                          % The transport block size
numTrBlks = 1;                     % Number of simulated transport blocks
ue = struct();                      % Initialize the UE structure
ue.NBULSubcarrierSpacing = '15kHz'; % 3.75kHz, 15kHz
ue.NNCellID = 10;                    % Narrowband cell identity
chs = struct();
% NPUSCH carries data or control information
chs.NPUSCHFormat = 'Data'; % Payload type (Data or Control)
% The number of subcarriers used for NPUSCH 'NscRU' depends on the NPUSCH
% format and subcarrier spacing 'NBULSubcarrierSpacing' as shown in TS
% 36.211 Table 10.1.2.3-1. There are 1,3,6 or 12 contiguous subcarriers for
% NPUSCH
chs.NBULSubcarrierSet = 0:11;  % Range is 0-11 (15kHz); 0-47 (3.75kHz)
chs.NRUsc = length(chs.NBULSubcarrierSet);
chs.CyclicShift = 0;   % Cyclic shift required when NRUsc = 3 or 6
chs.RNTI = 0;          % RNTI value
chs.NLayers = 1;       % Number of layers
chs.NRU = 1;           % Number of resource units
chs.NRep = 16;          % Number of repetitions of the NPUSCH
chs.SlotIdx = 0;       % Start slot index in a bundle
% The symbol modulation depends on the NPUSCH format and NscRU as
% given by TS 36.211 Table 10.1.3.2-1
chs.Modulation = 'QPSK';
rvDCI = 0;             % RV offset signaled via DCI (See 36.213 16.5.1.2)
% Specify the NPUSCH and DM-RS power scaling in dB for plot visualization
chs.NPUSCHPower = 30;
chs.NPUSCHDRSPower = 34;
chs.SeqGroupHopping = 'on'; % Enable/Disable Sequence-Group Hopping for UE
chs.SeqGroup = 0;           % Delta_SS. Higher-layer parameter groupAssignmentNPUSCH
% Get number of time slots in a resource unit NULSlots according to
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

NSlotsPerBundle = chs.NRU*chs.NULSlots*chs.NRep; % Number of slots in a codeword bundle
TotNSlots = numTrBlks*NSlotsPerBundle;   % Total number of simulated slots
emptySlotGrid = lteNBResourceGrid(ue);
slotGridSize = size(emptySlotGrid);
NSlotsPerFrame = 20/(slotGridSize(1)/12);
tSlot = 10e-3/NSlotsPerFrame; % Slot duration
symbolsPerSlot = slotGridSize(2); % Number of symbols per slot
ueInit = ue;
chsInit = chs;
channelInit = channel;
% RV offset signaled via DCI (See 36.213 16.5.1.2)
rvDCI = 0;
% Calculate the RVSeq used according to the RV offset
rvSeq = [2*mod(rvDCI+0,2)  2*mod(rvDCI+1,2)];
if strcmpi(chs.NPUSCHFormat,'Data')
    infoLen = length(data); %136;   % Transport block size for NPUSCH format 1
elseif strcmpi(chs.NPUSCHFormat,'Control')
    infoLen = 1;    % ACK/NACK bit for NPUSCH format 2
end
state = [];    % NPUSCH encoder and DM-RS state, auto re-initialization in the function
trblk = [];    % Initialize the transport block
txgrid = [];   % Full grid initialization

%% Satellite Geometry and Parameter
beamwidth = 80.5* pi/180;                 % Satellite antenna beamwidth
Gain = 2/(1-cos(beamwidth/2));          % Ideal direction antenna gain
%% Fixed Parameters
BW=15e3+(chs.NRUsc-1)*15e3;
fc  = 921.6e6 ;                      % fc of the observer
Ro  = earthRadius;
ho  = 550e3;                         % Constellation Altitude
T   = 290;                           % Noise Temperature
k   = physconst('Boltzmann');
NF  = 10^(6/10);                     % Noise Figure (6dB by default)
N0  = k*T*BW*NF*1000;
env = [0.4 0 1 5 2];                 % Envirnomental factor of StG pathloss (Please refer to ITU-R for different Scenairo) the default [0.4 0 1 5 2] is for remote area
r   = Ro + ho; % satellite orbital radius
Fc  = 921.5e6 ;                      % Carrier Freq of the observer
Fs  = 1920000 ;                      % NB-IoT Fs
%% Simulation PArameters
Servingcell = [];
WindowTime = 1 ;                     % This is in second
Nsamples = WindowTime * Fs ;
%% Parameter that we wants to optimise
Niot    = ceil(510.1e9 * 2*0.061 / (24*3600*64)) ;      % probablity of active user under the satellite N*NumofTxPerDay*packetlenght
Power   = [-10:2:-2 -1.9:0.1:2.9 3:2:30];%[-10:2:-2 -1.9:0.1:2 2:2:30];
itr=100;
%%
% task 1 : create random starting time -done
% task 2 : create random interference again
% task 3 : random tx time
% task 4 : add them tgt
% task 5 : repeat

for cnt = 1:itr
    clearvars DeltaX DeltaY DeltaZ Rk k axang Local_Sat el_iot az_iot x_iot y_iot z_iot RxIQ sumIQ outputIQ StartTxtime Servingcell ObserverTimeline
    Servingcell = []; StartTxtime=[];
    load('SatelliteData\SatelliteDuty400_N_480_inc_870_WalkerS'); % this is the wallker constellation .mat you created using SatelliteOnDutyCalculation.m
    %% pick up one device (Uniform Distribution)
    while numel(Servingcell) == 0
        ChosenObserver = randi(500,1);                          % one of the users in the script
        ObserverTimeline = OnDuty(:,ChosenObserver,:);         %the time when the observer being serve by any satellites
        [Servingcell,StartTxtime]=find(ObserverTimeline);       % give me the full table of
    end
    Serveidx=randi(numel(Servingcell),1);                       % start at random time within the accesible period
    %% Serving cell FSPL and excessivepathloss
    k = cross([x_iot(ChosenObserver),y_iot(ChosenObserver),z_iot(ChosenObserver)],...
        [0,0,1]);  % find the rotation axis
    axang = [k, (pi/2-el_iot(ChosenObserver))];        %rotate to
    Rk = axang2rotm(axang);
    
    Local_Sat = Rk*[x_sat(Servingcell(Serveidx),StartTxtime(Serveidx));...
        y_sat(Servingcell(Serveidx),StartTxtime(Serveidx)); z_sat(Servingcell(Serveidx),StartTxtime(Serveidx))];
    
    DeltaX = Local_Sat(1,:)-0;
    DeltaY = Local_Sat(2,:)-0;
    DeltaZ = Local_Sat(3,:)-1;      %doing another transformation
    
    [~,ServingEl,Servingslantdist] = cart2sph(DeltaX,DeltaY,DeltaZ);
    Servingslantdist = Servingslantdist*Ro;
    
    clearvars DeltaX DeltaY DeltaZ Rk k axang Local_Sat el_iot az_iot x_iot y_iot z_iot
    %% Create interferencer PPP
    el_iot = pi/2-acos(2*rand(Niot,1)-1); % These are with respect to ECI frame
    az_iot = 2*pi*rand(Niot,1);            %random point
    [x_iot,y_iot,z_iot] = sph2cart(az_iot,el_iot,1); % This is w.r.t. ECI
    %% Calculating Distance between iot and the sat
    %Nt x DutyCycle (510e6 (1 [units/km2])* DutyCycle (2/24*60*60)     %Nt cannot be too much
    for ctr=1:Niot
        k = cross([x_iot(ctr),y_iot(ctr),z_iot(ctr)],...
            [0,0,1]);  % find the rotation axis
        axang = [k, (pi/2-el_iot(ctr))];        %rotate to
        Rk = axang2rotm(axang);
        Local_Sat = Rk*[x_sat(Servingcell(Serveidx),StartTxtime(Serveidx));...
            y_sat(Servingcell(Serveidx),StartTxtime(Serveidx)); z_sat(Servingcell(Serveidx),StartTxtime(Serveidx))];
        
        DeltaX = Local_Sat(1,:)-0;
        DeltaY = Local_Sat(2,:)-0;
        DeltaZ = Local_Sat(3,:)-1;      %doing another transformation
        
        [~,El(:,ctr),slantdist(:,ctr)] = cart2sph(DeltaX,DeltaY,DeltaZ);
    end
    
    DistanceStD = slantdist.*Ro ;
    
    %% find the interferencing cell
    varphi = acos ( ((r^2 + Ro^2 - DistanceStD.^2)/ (2 * r * Ro)) );
    if beamwidth < 2*asin(alpha)
        varphi_max = asin(sin(beamwidth/2)/alpha)-beamwidth/2 ;
    else
        varphi_max =  acos(alpha) ;
    end
    interferencing_cell = varphi < varphi_max;
    %% Uniform distribution for activity
    %         fcn = Fc-Fs/2+ Fs*rand(1,sum(interferencing_cell) );           % choose random Channel
    %     randomstarttime = randi([-Samplelength,Samplelength],1,sum(interferencing_cell));
    [FSPL,Eta] = StG_PathLoss(ServingEl*180/pi,Servingslantdist,Fc,[0.4 0 1 5 2]);
    
    % Initialize BLER and throughput result
    maxThroughput = zeros(length(Power),1);
    simThroughput = zeros(length(Power),1);
    bler = zeros(1,numel(Power)); % Initialize BLER result
    
    for PowerIdx = 1:numel(Power)
        
        numBlkErrors = 0;  % Number of transport blocks with errors
        
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
                    [InterIQ,scfdmaInfo] = lteSCFDMAModulate(ue,chs,slotGrid);
                    % Pass data through channel model
                    channel.SamplingRate = scfdmaInfo.SamplingRate;
                    [InterIQ,fadingInfo] = lteFadingChannel(channel, InterIQ);
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
            %------------------------------------------------------------------
            %            Receiver
            %------------------------------------------------------------------
            
            % Perform timing synchronization, extract the appropriate
            % subframe of the received waveform, and perform SC-FDMA
            % demodulation
            if (perfectChannelEstimator)
                offset = 0;
                %                     hPerfectTimingEstimate(fadingInfo);
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
            noiseEst = var(noiseGrid(:));
            
            % Get NPUSCH indices
            npuschIndices = lteNPUSCHIndices(ue,chs);
            % Get NPUSCH resource elements from the received slot
            [rxNpuschSymbols, npuschHest] = lteExtractResources(npuschIndices, ...
                rxSlot, estChannelGrid);
            
            % Perform channel estimate and noise estimate buffering in
            % the case of practical channel estimation
            % Decode NPUSCH
            if isnan(noiseEst)
                noiseEst = 0;
            end
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
        % Calculate the block error rslotate
        bler(PowerIdx) = numBlkErrors/numTrBlks;
        %             fprintf('NPUSCH BLER = %.4f \n',bler(PowerIdx));
        % Calculate the maximum and simulated throughput
        maxThroughput(PowerIdx) = infoLen*numTrBlks; % Max possible throughput
        simThroughput(PowerIdx) = infoLen*(numTrBlks-numBlkErrors);  % Simulated throughput
        %             fprintf('NPUSCH Throughput(%%) = %.4f %%\n',simThroughput(PowerIdx)*100/maxThroughput(PowerIdx));
        clearvars DeltaX DeltaY DeltaZ Rk k axang Local_Sat el_iot az_iot x_iot y_iot z_iot interIQ sumIQ outputIQ NoiseIQ RxIQ
    end
    
    
    
    %% Packet Error Rate
    Bler(cnt,:)=bler(:);
    bler=[];
end
    
    
    clearvars DeltaX DeltaY DeltaZ Rk k axang Local_Sat el_iot az_iot x_iot y_iot z_iot interIQ sumIQ outputIQ NoiseIQ
    save('PERvsTxpowerCase3_NBIoT_simRep32');

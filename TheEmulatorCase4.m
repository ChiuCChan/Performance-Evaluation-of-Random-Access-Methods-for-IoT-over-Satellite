clear
clc

beamwidth = 80.9*pi/180;             % Satellite antenna beamwidth
Gain = 2/(1-cos(beamwidth/2)) ;      % Ideal directional antenna gain equation
%% Fixed Parameters
SF  = 7 ;                            % LoRa Spreading Factor
BW  = 125e3;                         % Bandwidth of LoRa signal
fc  = 921.6e6 ;                      % Carrier freq of the observer
Ro  = earthRadius;
ho  = 550e3;                         % Constellation Altitude
T   = 290;                           % Noise Temperature
k   = physconst('Boltzmann');
NF  = 10^(6/10);                     % Noise Figure (6dB by default)
N0  = k*T*BW*NF*1000;
env = [0.4 0 1 5 2];                 % Envirnomental factor of StG pathloss (Please refer to ITU-R for different Scenairo) the default [0.4 0 1 5 2] is for remote area
dt = 2;
%% Calculations
r   = Ro + ho; % satellite orbital radius
%% LoRa Packet Parameters
message = "Hello World!";            % Msg content
Fs  = 10e6;                          % Sampling Freq
Fc  = 921.5e6 ;                      % Carrier Freq of LoRa
%% Simulation PArameters
Servingcell = [];
WindowTime = 1 ;                     % This is in second
Nsamples = WindowTime * Fs ;         % Num of Samples
%% Parameter that we wants to optimise
Niot    = ceil(510.1e9 * 2*0.061 / (24*3600*64)) ;       % probablity of active user under the satellite N*NumofTxPerDay*packetlenght
Power   = 10:2:40                    % Tx power of observer
pI  = 10:2:40;                       % Tx power of interference
itr=100;
for pwrcnt=1:numel(Power)
    for cnt = 1:itr
        clearvars DeltaX DeltaY DeltaZ Rk k axang Local_Sat el_iot az_iot x_iot y_iot z_iot sumIQ outputIQ
        load('SatelliteData\SatelliteDuty400_N_480_inc_875_WalkerS'); % this is the wallker constellation you created using SatelliteOnDutyCalculation.m

        %% pick up one device (Uniform Distribution)
        while numel(Servingcell) == 0
            ChosenObserver = randi(Nt,1);                          % one of the users in the script
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
        clearvars DeltaX DeltaY DeltaZ Rk k axang
        %% Add doppler
        for dopplercnt = 1:5
            k = cross([x_iot(ChosenObserver),y_iot(ChosenObserver),z_iot(ChosenObserver)],...
                [0,0,1]);  % find the rotation axis
            axang = [k, (pi/2-el_iot(ChosenObserver))];        %rotate to
            Rk = axang2rotm(axang);
            if StartTxtime(Serveidx)<= 1;
                Local_Sat = Rk*[x_sat(Servingcell(Serveidx),ceil(StartTxtime(Serveidx)-1+dopplercnt))';...
                    y_sat(Servingcell(Serveidx),ceil(StartTxtime(Serveidx)-1+dopplercnt))'; z_sat(Servingcell(Serveidx),ceil(StartTxtime(Serveidx)-1+dopplercnt))'];
            elseif StartTxtime(Serveidx)> 5540;
                Local_Sat = Rk*[x_sat(Servingcell(Serveidx),ceil(StartTxtime(Serveidx)-5+dopplercnt))';...
                    y_sat(Servingcell(Serveidx),ceil(StartTxtime(Serveidx)-5+dopplercnt))'; z_sat(Servingcell(Serveidx),ceil(StartTxtime(Serveidx)-5+dopplercnt))'];
            else
                Local_Sat = Rk*[x_sat(Servingcell(Serveidx),ceil(StartTxtime(Serveidx)-3+dopplercnt))';...
                    y_sat(Servingcell(Serveidx),ceil(StartTxtime(Serveidx)-3+dopplercnt))'; z_sat(Servingcell(Serveidx),ceil(StartTxtime(Serveidx)-3+dopplercnt))'];
            end

            DeltaX = Local_Sat(1,:)-0;
            DeltaY = Local_Sat(2,:)-0;
            DeltaZ = Local_Sat(3,:)-1;      %doing another transformation

            [~,ServingEl,dopplerslantdist(dopplercnt)] = cart2sph(DeltaX,DeltaY,DeltaZ);
            dopplerslantdist(dopplercnt) = dopplerslantdist(dopplercnt)*Ro;
            clearvars DeltaX DeltaY DeltaZ Rk k axang
        end
        diffd=mean(diff(dopplerslantdist)./dt);
        fd=-diffd*(fc/3e8);
        clearvars DeltaX DeltaY DeltaZ Rk k axang Local_Sat el_iot az_iot x_iot y_iot z_iot diffd
        %% Create interferencer PPP
        el_iot = pi/2-acos(2*rand(Niot,1)-1); % These are with respect to ECI frame
        az_iot = 2*pi*rand(Niot,1);            %random point
        [x_iot,y_iot,z_iot] = sph2cart(az_iot,el_iot,1); % This is w.r.t. ECI
        %% Calculating Distance between iot and the sat
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
        varphi = acos ( ((r^2 + Ro^2 - DistanceStD.^2)/ (2 * r * Ro)) );
        [FSPL,Eta] = StG_PathLoss(ServingEl*180/pi,Servingslantdist,Fc,[0.4 0 1 5 2]);
        signalIQ = LoRa_Tx(message,BW,SF,Power(pwrcnt),Fs,Fc - fc);
        time = (0:numel(signalIQ)-1)'.*1/Fs;
        signalIQ = signalIQ.*exp(j*2*pi.*fd.*time);
        ObserverPacketLength = length(signalIQ);
        outputIQ = signalIQ.*10^( -(FSPL + Eta) ./20);
        outputIQ = [zeros(1,2e6) outputIQ'] ;
        sumIQ    = zeros(1,Nsamples) ;
        %%
        if length(outputIQ) < length(sumIQ)
            outputIQ = [outputIQ zeros(1,length(sumIQ)-length(outputIQ))] ;
        elseif length(outputIQ) > length(sumIQ)
            error('Increase Time Window')
        end
        sumIQ       = sumIQ + outputIQ ;

        RxIQ =  sumIQ(2e6+1:2e6+1+ObserverPacketLength);
        RxIQ = resample(RxIQ,BW,Fs) ;
        NoiseIQ = sqrt(N0/2).*((randn(1,length(RxIQ))) + j.*randn(1,length(RxIQ))) ;
        RxIQ = RxIQ.*Gain;
        RxIQ = RxIQ + NoiseIQ;
        message_out = LoRa_Rx(RxIQ',BW,SF,2,BW,Fc - fc) ;
        %% Packet Error Rate
        if (sum(isnan(message_out))||sum(isempty(message_out)))
            SUC(cnt,:)=zeros(1,length(char(message)));
        elseif numel(message_out)<numel(char(message))
            SUC(cnt,:)=zeros(1,length(char(message)));
        else
            SUC(cnt,:)=eq(char(message_out(1:length(char(message)))), char(message));
        end

    end
    PERrate(pwrcnt) = 1 - (sum(sum(SUC))/numel(SUC));
    clear SUC
end
clearvars DeltaX DeltaY DeltaZ Rk k axang Local_Sat el_iot az_iot x_iot y_iot z_iot interIQ sumIQ outputIQ NoiseIQ
save('PERvsTxpowerCase4');

clear
clc

beamwidth = 80.9* pi/180;
Gain = 2/(1-cos(beamwidth/2));
%%
SF = 7 ;                              % LoRa Spreading Factor
BW = 125e3 ;                          % Bandwidth of LoRa signal
fc = 921.6e6 ;                        % fc of the observer
Ro=earthRadius;
ho = 550e3;                           % constellation altitude
T = 290;                              % Noise Temperature
k = physconst('Boltzmann');
NF=10^(6/10);                         % Noise figure (6dB by default)
N0=k*T*BW*NF*1000;                    
env=[0.4 0 1 5 2];                    % Envirnomental factor of StG pathloss (Please refer to ITU-R for different Scenairo) the default [0.4 0 1 5 2] is for remote area
dt = 2;                               % observation window step
itr=100;                              % Num of Iteration
%% Calculations
r  = Ro + ho; % satellite orbital radius
%% LoRa Packet Parameters
message = "Hello World!" ;          % Msg Content
Fs = 10e6 ;                         % Sample freq
Fc = 921.5e6 ;                      % Carrier Freq of LoRa
%% Simulation PArameters
Servingcell=[];
WindowTime = 1;                     % This is in second
Nsamples = WindowTime * Fs;         % Num of Samples
%% Parameter that we wants to optimise
Niot = ceil(510.1e9 * 2*0.061 / (24*3600*64));       % probablity of active user under the satellite N*NumofTxPerDay*packetlenght
Power =[10:2:40];                   % Tx power of observer
pI = [10:2:40];                     % Tx power of Interference

for pwrcnt=1:numel(Power)
    for cnt = 1:itr
        clearvars DeltaX DeltaY DeltaZ Rk k axang Local_Sat el_iot az_iot x_iot y_iot z_iot sumIQ outputIQ
        load('SatelliteData\SatelliteDuty400_N_480_inc_875_WalkerS');
        %% pick up one device (Uniform Distribution)
        while numel(Servingcell) == 0;
            ChosenObserver = randi(Nt,1);      % choose one of the users in the dataset
            ObserverTimeline = Dutying(:,ChosenObserver,:);         % the time period when the observer being serve by any satellites
            [Servingcell,StartTxtime]=find(ObserverTimeline);       % finding the servingcell and its corresponding serving time(s)
        end
        Serveidx=randi(numel(Servingcell),1);                       % Randomly pick up one Serving cell and time
        %% Serving cell FSPL and excessivepathloss
        % find slantdis and El angle from observer to all satellites at the observation time
        k = cross([x_iot(ChosenObserver),y_iot(ChosenObserver),z_iot(ChosenObserver)],...
            [0,0,1]);  % find the rotation axis
        axang = [k, (pi/2-el_iot(ChosenObserver))];
        Rk = axang2rotm(axang);

        Local_Sat = Rk*[x_sat(Servingcell(Serveidx),StartTxtime(Serveidx));...
            y_sat(Servingcell(Serveidx),StartTxtime(Serveidx)); z_sat(Servingcell(Serveidx),StartTxtime(Serveidx))];

        DeltaX = Local_Sat(1,:)-0;
        DeltaY = Local_Sat(2,:)-0;
        DeltaZ = Local_Sat(3,:)-1;   

        [~,ServingEl,Servingslantdist] = cart2sph(DeltaX,DeltaY,DeltaZ);
        Servingslantdist = Servingslantdist*Ro;
        %% doppler
        for dopplercnt = 1:5
            k = cross([x_iot(ChosenObserver),y_iot(ChosenObserver),z_iot(ChosenObserver)],...
                [0,0,1]);  % find the rotation axis
            axang = [k, (pi/2-el_iot(ChosenObserver))];       
            Rk = axang2rotm(axang);
            if StartTxtime(Serveidx)== 1
                Local_Sat = Rk*[x_sat(Servingcell(Serveidx),ceil(StartTxtime(Serveidx)-1+dopplercnt))';...
                    y_sat(Servingcell(Serveidx),ceil(StartTxtime(Serveidx)-1+dopplercnt))'; z_sat(Servingcell(Serveidx),ceil(StartTxtime(Serveidx)-1+dopplercnt))'];
            elseif StartTxtime(Serveidx)== size(x_sat,2)
                Local_Sat = Rk*[x_sat(Servingcell(Serveidx),ceil(StartTxtime(Serveidx)-5+dopplercnt))';...
                    y_sat(Servingcell(Serveidx),ceil(StartTxtime(Serveidx)-5+dopplercnt))'; z_sat(Servingcell(Serveidx),ceil(StartTxtime(Serveidx)-5+dopplercnt))'];
            else
                Local_Sat = Rk*[x_sat(Servingcell(Serveidx),ceil(StartTxtime(Serveidx)-3+dopplercnt))';...
                    y_sat(Servingcell(Serveidx),ceil(StartTxtime(Serveidx)-3+dopplercnt))'; z_sat(Servingcell(Serveidx),ceil(StartTxtime(Serveidx)-3+dopplercnt))'];
            end
            DeltaX = Local_Sat(1,:)-0;
            DeltaY = Local_Sat(2,:)-0;
            DeltaZ = Local_Sat(3,:)-1;     

            [~,ServingEl,dopplerslantdist(dopplercnt)] = cart2sph(DeltaX,DeltaY,DeltaZ);
            dopplerslantdist(dopplercnt) = dopplerslantdist(dopplercnt)*Ro;
            clearvars DeltaX DeltaY DeltaZ Rk k axang
        end
        diffd=mean(diff(dopplerslantdist)./dt);
        fd=-diffd*(fc/3e8);
        clearvars DeltaX DeltaY DeltaZ Rk k axang Local_Sat el_iot az_iot x_iot y_iot z_iot diffd
        %% Create interferencer PPP on the sphere

        el_iot = pi/2-acos(2*rand(Niot,1)-1); % These are with respect to ECI frame
        az_iot = 2*pi*rand(Niot,1);            %random point
        [x_iot,y_iot,z_iot] = sph2cart(az_iot,el_iot,1); % This is w.r.t. ECI
        %% Calculating Distance between iot and the sat
        % find slantdis and El angle from IoST devices to all satellites at the observation time
        for ctr=1:Niot
            k = cross([x_iot(ctr),y_iot(ctr),z_iot(ctr)],...
                [0,0,1]);  % find the rotation axis
            axang = [k, (pi/2-el_iot(ctr))];      
            Rk = axang2rotm(axang);
            Local_Sat = Rk*[x_sat(Servingcell(Serveidx),StartTxtime(Serveidx));...
                y_sat(Servingcell(Serveidx),StartTxtime(Serveidx)); z_sat(Servingcell(Serveidx),StartTxtime(Serveidx))];

            DeltaX = Local_Sat(1,:)-0;
            DeltaY = Local_Sat(2,:)-0;
            DeltaZ = Local_Sat(3,:)-1;     

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

        for rep = 1:4
            randomstarttime = randi(Nsamples,1,sum(interferencing_cell));
            %     randomstarttime = randi([-Samplelength,Samplelength],1,sum(interferencing_cell));
            [FSPL,Eta] = StG_PathLoss(ServingEl*180/pi,Servingslantdist,Fc,[0.4 0 1 5 2]);
            signalIQ = LoRa_Tx(message,BW,SF,Power(pwrcnt),Fs,Fc - fc);
            time = (0:numel(signalIQ)-1)'.*1/Fs;
            signalIQ = signalIQ.*exp(j*2*pi.*fd.*time);
            ObserverPacketLength = length(signalIQ) ;
            outputIQ = signalIQ.*10^( -(FSPL + Eta) ./20);
            outputIQ = [zeros(1,2e6) outputIQ'] ;
            sumIQ    = zeros(1,Nsamples) ;
            if sum(interferencing_cell) > 0
                [FSPL_int,Eta_int] = StG_PathLoss(El(interferencing_cell)*180/pi,DistanceStD(interferencing_cell),fc,[0.4 0 1 5 2]) ;
                for n=1:sum(interferencing_cell)
                    %% this part is calculating the interference signal
                    InterIQ = LoRa_Tx(message,BW,SF,pI(pwrcnt),Fs,Fc - fc);
                    InterIQ = InterIQ.*10^( -(FSPL_int(n) + Eta_int(n)) ./ 20);
                    InterIQ     = [zeros(1,randomstarttime(n)-1) InterIQ'] ;
                    if length(InterIQ) < length(sumIQ)
                        InterIQ = [InterIQ zeros(1,length(sumIQ)-length(InterIQ))] ;
                    elseif length(InterIQ) > length(sumIQ)
                        InterIQ = InterIQ(1:Nsamples) ;

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

            RxIQ =  sumIQ(2e6+1:2e6+1+ObserverPacketLength);
            RxIQ = resample(RxIQ,BW,Fs) ;
            NoiseIQ = sqrt(N0/2).*((randn(1,length(RxIQ))) + j.*randn(1,length(RxIQ))) ;
            RxIQ = RxIQ.*Gain;
            RxIQ = RxIQ + NoiseIQ;

            message_out = LoRa_Rx(RxIQ',BW,SF,2,BW,Fc - fc) ;

            %% Packet Error Rate
            if (sum(isnan(message_out))||sum(isempty(message_out)))
                if rep == 1
                    SUC(cnt,:) = zeros(1,length(char(message)));
                else
                    SUC(cnt,:) = SUC(cnt,:)|zeros(1,length(char(message)));
                end
            elseif numel(message_out)<numel(char(message))
                if rep == 1
                    SUC(cnt,:) = zeros(1,length(char(message)));
                else
                    SUC(cnt,:) = SUC(cnt,:)|zeros(1,length(char(message)));
                end
            else
                if rep == 1
                    SUC(cnt,:) = eq(char(message_out(1:length(char(message)))), char(message));
                else
                    SUC(cnt,:) = SUC(cnt,:)|eq(char(message_out(1:length(char(message)))), char(message));
                end
            end
            if sum(SUC(cnt,:)) == numel(char(message)) %    if PER = 0, then break
                break
            else
                clearvars interIQ sumIQ outputIQ NoiseIQ
            end
        end

    end
    PERrate(pwrcnt) = 1- sum(sum(SUC(:,:))) / ( itr * numel(char(message)) ); %PER = 1 - suc bit / total bit
    clear SUC
end
clearvars DeltaX DeltaY DeltaZ Rk k axang Local_Sat el_iot az_iot x_iot y_iot z_iot interIQ sumIQ outputIQ NoiseIQ SUC
save('PERvsTxpowerCase3_rep4');

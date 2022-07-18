clear
clc

%%
SF = 7 ;                            % LoRa Spreading Factor
BW = 125e3 ;                        % Bandwidth of LoRa signal
fc = 921.6e6 ;                      % Carrier freq of the observer
beamwidth = 80.9* pi/180;           % Satellite antenna beamwidth
Ro=earthRadius;
Gain = 2/(1-cos(beamwidth/2));      % Ideal directional antenna gain equation
T = 290;                            % NoiseTemp
k = physconst('Boltzmann');
NF=10^(6/10);                       % Noise Figure (6dB by default)
N0=k*T*BW*NF*1000;
h = 550e3;                          % constellation altitude
r  = Ro + h; % satellite orbital radius

message = "Hello World!" ;          % Msg content
message_out=[];
env=[0.4 0 1 5 2];                  % Envirnomental factor of StG pathloss (Please refer to ITU-R for different Scenairo) the default [0.4 0 1 5 2] is for remote area
Fs = 10e6 ;                         % Sampling Freq
Fc = 921.5e6 ;                      % Carrier Freq of LoRa
Servingcell=[];
dt = 2;                             % observation window step
WindowTime = 1;                     % This is in second
Nsamples = WindowTime * Fs;         % Num of Samples
itr= 100;                           % Num of iteration
%% Parameter that we wants to optimise
Niot = ceil(510.1e9 * 2*0.061 / (24*3600*64));       % probablity of active user under the satellite N*NumofTxPerDay*packetlenght
Power = [10:2:40];
PI = [10:2:40];                     % Tx power for everyone else
endtime = 2000;                     % Start Tx time (end time has to be less than the size of 3 column of Onduty)
StartTxtime = linspace(1,endtime,itr); 
%%
% task 1 : create random starting time -done
% task 2 : create random interference again
% task 3 : random tx time
% task 4 : add them tgt
% task 5 : repeat

%% Creating Tx
Obs_az = 2*pi*rand(1);
  Obs_el = pi/2; % These are with respect to ECI frame with 90 inc
%  Obs_el = [0:5:90].* pi/180; % These are with respect to ECI frame with 60 inc
%% creating traffic
for pwrcnt=1:numel(Power)
    for inccnt = 1:numel(Obs_el)
        for cnt = 1:itr
            clearvars DeltaX DeltaY DeltaZ Rk k axang Local_Sat el_iot az_iot x_iot y_iot z_iot sumIQ outputIQ DeltaX_Obs DeltaY_Obs DeltaZ_Obs
            load('SatelliteData\SatelliteDuty800_N_1200_inc_875_WalkerS'); % this is the wallker constellation .mat you created using SatelliteOnDutyCalculation.m

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
            clearvars DeltaX_Obs DeltaY_Obs DeltaZ_Obs Rk k axang
            %%
            [~,ServingCell] = min(DistanceStD_Obs) ;                  %find the serving satellite
            Servingslantdist=DistanceStD_Obs(ServingCell);
            ServingEl=El_Obs(ServingCell);
            %% Add doppler
            for dopplercnt = 1:5
                k = cross([x_Obs,y_Obs,z_Obs],...
                    [0,0,1]);  % find the rotation axis
                axang = [k, (pi/2-Obs_el(inccnt))];        %rotate to
                Rk = axang2rotm(axang);
                if StartTxtime(cnt)== 1
                    Local_Sat = Rk*[x_sat(ServingCell,ceil(StartTxtime(cnt)-1+dopplercnt))';...
                        y_sat(ServingCell,ceil(StartTxtime(cnt)-1+dopplercnt))'; z_sat(ServingCell,ceil(StartTxtime(cnt)-1+dopplercnt))'];
                elseif StartTxtime(cnt)== size(x_sat,2)
                    Local_Sat = Rk*[x_sat(ServingCell,ceil(StartTxtime(cnt)-5+dopplercnt))';...
                        y_sat(ServingCell,ceil(StartTxtime(cnt)-5+dopplercnt))'; z_sat(ServingCell,ceil(StartTxtime(cnt)-5+dopplercnt))'];
                else
                    Local_Sat = Rk*[x_sat(ServingCell,ceil(StartTxtime(cnt)-3+dopplercnt))';...
                        y_sat(ServingCell,ceil(StartTxtime(cnt)-3+dopplercnt))'; z_sat(ServingCell,ceil(StartTxtime(cnt)-3+dopplercnt))'];
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
            clearvars DeltaX DeltaY DeltaZ Rk k axang Local_Sat diffd
            %%
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
            if ServingEl > 0
                if beamwidth < 2*asin(alpha)
                    varphi_max = asin(sin(beamwidth/2)/alpha)-beamwidth/2 ;
                else
                    varphi_max =  acos(alpha) ;
                end
                if varphi_Obs < varphi_max
                    interferencing_cell = varphi < varphi_max;
                    %         fcn = Fc-Fs/2+ Fs*rand(1,sum(interferencing_cell) );           % choose random Channel
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
                            InterIQ = LoRa_Tx(message,BW,SF,PI(pwrcnt),Fs,Fc - fc);
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
                        SUC(cnt,:)=zeros(1,length(char(message)));
                    elseif numel(message_out)<numel(char(message))
                        SUC(cnt,:)=zeros(1,length(char(message)));
                    else
                        SUC(cnt,:)=eq(char(message_out(1:length(char(message)))), char(message) );
                    end
                else
                    SUC(cnt,:)=zeros(1,length(char(message)));
                end
            else
                SUC(cnt,:)=zeros(1,length(char(message)));
            end
        end
        SUC_temp(inccnt,:)=SUC(:)';
        clear SUC
    end
    PERrate(pwrcnt) = 1 - (sum(sum(SUC_temp))/numel(SUC_temp));
    clear SUC SUC_temp
end

clearvars DeltaX DeltaY DeltaZ Rk k axang Local_Sat el_iot az_iot x_iot y_iot z_iot interIQ sumIQ outputIQ NoiseIQ
save('PERvsTxpowerCase1');

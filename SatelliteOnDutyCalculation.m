clc
clear
close all
%% Walker paramters

% These are Space-X paramters thrid revision
%N   = 1584;
%inc = 53 *pi/180;% orbital inclincation
%P   = 72; % number of orbital planes
%F   = 1*N/P;
%h   = 550e3; % satellite height
% WalkerDelta = 1;
%Alt = 0; % This is logical paramters to alternate the rotation diractions betwen consequative planes

% These are Space-X paramters first revision
% N   = 1600;
% inc = 53 *pi/180;% orbital inclincation
% P   = 32 ; % number of orbital planes
% F   = 1*N/P;
% h   = 1150e3; % satellite height
% WalkerDelta = 1;
% Alt = 0; % This is logical paramters to alternate the rotation diractions betwen consequative planes

% % For plotting Irriduim
% N   = 66;
% inc = 90 *pi/180;% orbital inclincation
% P   = 6 ; % number of orbital planes
% F   = 1*N/P;
% h   = 400e3; % satellite height
% WalkerDelta = 0;
% Alt = 0; % This is logical paramters to alternate the rotation diractions betwen consequative planes


% For plotting
N   = 480;  %1200
inc = 40.*pi/180;% orbital inclincation
P   = 16 ; % number of orbital planes
F   = P/2;
h   = 550e3; % satellite height
WalkerDelta = 1;
Alt = 0; % This is logical paramters to alternate the rotation diractions betwen consequative planes

% walker star
% remove the user from the (inc > 15n deploy  \circ )

% Paramters
Nt=400;
Ro = earthRadius;
r  = Ro + h; % satellite orbital radius
rho_sat = r/ Ro;
alpha = Ro/ (h + Ro);
mu = 3.986e14; % Erth's standard gravitational parameter
omega  = sqrt(mu/r.^3); % Angular velocity
T  = 2*pi*sqrt(r.^3/mu);
dt = 2; %time step
t  = 0:dt:T;% Time vector
beamwidth= 80.9 * (pi/180);

%% Generate the orbits
ctr_n=1; % Satellite index

for ctr_p =1:P
    if WalkerDelta
        % this is the case of Waleker-Delta
        RAAN = 2*pi/P *(ctr_p-1); % this is the Right ascention of the accending node
    else
        % this is for pure polar
        RAAN = pi/P *(ctr_p-1); % this is the Right ascention of the accending node
    end
    for ctr_s = 1:N/P % this will repeat for each satellite in every orbital plane
        % here we apply mean anomaly on each satellties
        % by cyclicly rotating the orbit to shift the starting position
        M = 2*pi/(N/P)*(ctr_s-1)+2*pi/N*F*(ctr_p-1);
        % generate the basic circlular orbit aligned with the x-y plane
        if Alt==1
            Sign = 2*mod(ctr_p,2)-1;
            BasicOrbit = [cos(Sign*omega*t+ M)*r/Ro; sin(Sign*omega*t+ M)*r/Ro; zeros(1,length(t))];
        else
            BasicOrbit = [cos(omega*t+ M)*r/Ro; sin(omega*t+ M)*r/Ro; zeros(1,length(t))];
        end
        % apply the inclincation
        axang = [[1 0 0],inc];
        Rk = axang2rotm(axang);
        Orbit = Rk*BasicOrbit;
        % rotate the starting point of the orbit to match the RAAN of orbital plane
        axang = [[0 0 1],RAAN];
        Rz = axang2rotm(axang);
        Orbit = Rz*Orbit;
        xOrbit_fix(ctr_n,:) = Orbit(1,:);
        yOrbit_fix(ctr_n,:) = Orbit(2,:);
        zOrbit_fix(ctr_n,:) = Orbit(3,:);
        ctr_n=ctr_n+1;
    end
end
%% Checking the constellation

az_Earth = 2*pi/(3600*24)*t; %Earth azimuth

for ctr_n=1:N % obtain the azimuth and eelvatiom of satellites
    [azSat(ctr_n,:),elZat(ctr_n,:),~]=cart2sph(xOrbit_fix(ctr_n,:),yOrbit_fix(ctr_n,:),zOrbit_fix(ctr_n,:));
    azSat(ctr_n,:)=azSat(ctr_n,:)-az_Earth;

end

%% Create IoT PPP
% el_sat = pi/2-acos(2*rand(N,numel(t))-1); % These are with respect to ECI frame
%         %         el_sat = pi/2-acos( 0.8660 *(2*rand(N,1)-1)); % These are with respect to ECI frame with 60 inc
%         %         el_sat = pi/2-acos( 0.6428 *(2*rand(N,1)-1)); % 40 inc
%         az_sat = 2*pi*rand(N,numel(t));            %random point
el_iot = pi/2-acos(2*rand(Nt,1)-1); % These are with respect to ECI frame
az_iot = 2*pi*rand(Nt,1);            %random point
[x_sat,y_sat,z_sat] = sph2cart(azSat,elZat,rho_sat); % this is for WD WS
[x_iot,y_iot,z_iot] = sph2cart(az_iot,el_iot,1); % This is w.r.t. ECI

%% Calculating Distance between iot and sat
for ctr=1:Nt
    k = cross([x_iot(ctr),y_iot(ctr),z_iot(ctr)],...
        [0,0,1]);  % find the rotation axis
    axang = [k, (pi/2-el_iot(ctr))];        %rotate to
    Rk = axang2rotm(axang);
    for tcnt = 1:numel(t)
        Local_Sat = Rk*[x_sat(:,tcnt)'; y_sat(:,tcnt)'; z_sat(:,tcnt)'];
        DeltaX = Local_Sat(1,:)-0;
        DeltaY = Local_Sat(2,:)-0;
        DeltaZ = Local_Sat(3,:)-1;      %doing another transformation
        [~,El(:,ctr,tcnt),slantdist(:,ctr,tcnt)] = cart2sph(DeltaX,DeltaY,DeltaZ);
    end
    %         varphi(:,ctr,tcnt) = acos ((r^2 + Ro^2 - slantdist(:,ctr,tcnt).^2)/ (2 * r * Ro));
end
%% Calculate the Phi_max
slantdist = slantdist.*Ro;
[minslantdist, servingcellind] = min(slantdist);
varphi = acos ( ((r^2 + Ro^2 - slantdist.^2)/ (2 * r * Ro)) );
varphi_max=min([ acos(alpha) asin(sin(beamwidth/2)/alpha)-beamwidth/2]);
OnDuty = varphi < varphi_max; %boolean of satellite s

%% Find the true serving cell
Dutytable=sum(OnDuty,1);        % sum all OnDuty satellites
temp=boolean(Dutytable);         % obtain N% union at instant t
Ntime=sum(temp,3);          % obtain total serving time N
%%
clearvars -except xOrbit_fix yOrbit_fix zOrbit_fix h inc P F N Alt Ro t dt T ...
    azSat elZat  ColorOrder Nt rho_sat r alpha beamwidth Ntime el_iot az_iot...
    x_sat y_sat z_sat x_iot y_iot z_iot OnDuty minslantdist servingcellind
%
%%
save(['SatelliteData\SatelliteDuty',num2str(h/1e3),'_N_',num2str(N),'_inc_',num2str(inc.*1800/pi),'_WalkerD']);

%%
% Programmed by Chiu Chan on Jan 2022
% (E-mail: chui.chan@rmit.edu.au/ChiuChun.Chan@anu.edu.au)
function [FSPL,Eta] = StG_PathLoss(El,Distance,Freq,x)
% This fcuntion produces a random path-loss + shadwogin component for a Satellite-to-Ground channel
% Inputs:
% El: The elevation angle of the satellites in degrees
% Distance: to the satellite in [m]
% Freq: Frequency in [Hz]
% x: is the model vector, where x(1) is beta, [x(2), x(3)] mean and std of
% the LoS component, [x(4), x(5)] mean and std of the NLoS component
% For more details on this model see the paper:
% A. Al-Hourani and I. Guvenc, “On modeling satellite-to-ground pathloss in urban environments,” Submitted to IEEE Wireless Communications Letters, pp. 1–1, 2020.
% Usage example:
% StG_PathLoss(45,800e3,915e6, [0.31    0    2   10   8])

PLoS = exp(-x(1)*cotd(El)); % LoS porbability

% Generate Bernouli selector
B = rand(size(El)) > (1-PLoS); % 1 is LoS and 0 is NLoS

Eta =  ( x(2)  + x(3)* randn(size(El))).*B + ... % los component
    ( x(4)  + x(5)* randn(size(El))).*(1-B); %nlos component
FSPL = 20*log10(Freq)+20*log10(Distance)-147.55;
FSPL(El<0)=inf;
Eta(El<0) =inf;
%Loss = FSPL + Eta;
end


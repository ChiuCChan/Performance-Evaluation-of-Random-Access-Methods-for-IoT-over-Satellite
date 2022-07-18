%hPerfectTimingEstimate Perfect timing estimation
%   [TOFFSET,MAG] = hPerfectTimingEstimate(CHANNELINFO) 
%   calculates the perfect timing estimate TOFFSET and the magnitude of the
%   impulse response MAG for the fading channel configuration information
%   specified in CHANNELINFO. CHANNELINFO is typically obtained as the
%   second output from the <a 
%   href="matlab:help('lteFadingChannel')">lteFadingChannel</a> function used in the simulation.

% Copyright 2017-2019 The MathWorks, Inc.

function [offset,mag] = hPerfectTimingEstimate(channelInfo)

    % Get number of receive antennas 'R' in the path gains array
    R = size(channelInfo.PathGains,4);
    
    % Establish integer path sample indices 'n'
    n = floor(channelInfo.PathSampleDelays) + 1;
    
    % Establish fractional path delays 'rho'
    rho = channelInfo.PathSampleDelays + 1 - n;
    
    % Create path filters by delaying by integer path sample index n(l) and
    % performing a first-order interpolation between two sample indices
    % according to the fractional path delay rho(l)
    L = numel(n);
    pathFilters = zeros(L,max(n)+1);
    for l = 1:L
        pathFilters(l,n(l) + [0;1]) = [(1-rho(l)); rho(l)];
    end
    
    % Compute the channel impulse response magnitude for each receive 
    % antenna
    [~,mag] = channelDelay(channelInfo.PathGains,pathFilters);
    
    % Return the minimum timing offset across the receive antennas that
    % have a correlation peak at least 50% of the magnitude of the
    % strongest peak, including accounting for the channel filter
    % implementation delay
    offset = zeros(1,R);
    maxmag = zeros(1,R);
    for r = 1:R
        maxmag(r) = max(mag(:,r));
        offset(r) = find(mag(:,r)==maxmag(r),1) - 1;
    end
    offset = min(offset(maxmag>=0.5*max(maxmag)));
    offset = offset + channelInfo.ChannelFilterDelay;

end

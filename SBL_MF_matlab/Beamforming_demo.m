%function Beamforming_demo()
% Beamforming_demo demonstrates capabilities of the SBL_v3p12.m code with
% single- or multiple-frequencies and multiple snapshots.
%
%
% Environment parameters
c      = 1500;       % speed of sound
f      = [200 250 300 350 400];  % frequency
%f= 200:-40:80%f      = [200 ];  % frequency
%f=200
lambda = c./f;       % wavelength

% ULA-horizontal array configuration
Nsensor = 20;                  % number of sensors
d       = 1/2*lambda(1);       % intersensor spacing for 200 Hz
q       = (0:1:(Nsensor-1))';  % sensor numbering
xq      = (q-(Nsensor-1)/2)*d; % sensor locations

% sensor configuration structure
Sensor_s.Nsensor = Nsensor;
Sensor_s.lambda  = lambda;
Sensor_s.Nlambda = length(lambda);
Sensor_s.d  = d;
Sensor_s.q  = q;
Sensor_s.xq = xq;

% SBL options structure for version 3
options = SBLSet();

% signal generation parameters
SNR = 40; -10 ; 10;

% total number of snapshots
Nsnapshot =   1;

% number of random repetitions to calculate the average performance
Nsim = 1;

% range of angle space
thetalim         = [-90 90];
theta_separation = 0.5;

% Bearing grid
theta  = (thetalim(1):theta_separation:thetalim(2))';
Ntheta = length(theta);

% Design/steering matrix
sin_theta = sind(theta);

% Multi-F dictionary / replicas
A = zeros(Nsensor,Ntheta,Sensor_s.Nlambda);
for iF = 1 : Sensor_s.Nlambda
    A(:,:,iF) = exp(-1i*2*pi/lambda(iF)*xq*sin_theta.')/sqrt(Nsensor);
end

% multiple simulations
for isim = 1:Nsim 
    Signal_s= generate_signal(Sensor_s, Nsnapshot, SNR, isim);
    Ysignal=Signal_s.Ysignal;
    % run CBF
    disp('Running CBF code');
    tic;
    K = zeros(Nsensor,Nsensor,Sensor_s.Nlambda);
    CBF = zeros(Ntheta,Sensor_s.Nlambda);
    
    for iF = 1 : Sensor_s.Nlambda
        YF=squeeze(Ysignal(:,:,iF));
        K(:,:,iF) = YF * YF'/Nsnapshot;
        CBF(:,iF) = real(diag(conj(squeeze(A(:,:,iF)).') * squeeze(K(:,:,iF))* squeeze(A(:,:,iF))));   
    end
    toc;
    
    % run SBL
    %%
    disp('Running SBL code');
    tic;
options.gamma_range=10^-4;
options.fixedpoint = 1;
options.convergence.error   = 10^(-4);
options.status_report = 1;
options.convergence.min_iteration = 1; 
options.Nsource = 3;
[gamma, SBLreport] = SBL_v4( A, Ysignal, options );
   
    toc;
    %%
    % run CVX,
    %     display('Running CVX code');
    %     mu = 70;    % regularization parameter
    %     tic;
    %     cvx_begin
    %         variable x_l1(Ntheta, Nsnapshot) complex;
    %         cvx_quiet(true)
    %         minimize( square_pos(norm(A * x_l1 - Ysignal, 'fro')) + mu * sum(norms(x_l1,2,2)) ) ;
    %     cvx_end
    %     toc;
    
    %%
    figure; clf
    
    % Single Freq CBF
    subplot(2,2,1);
    plot(Signal_s.source_theta(1), -5, 'rx', 'Markersize', 20);
    leg{1} = 'True source location';
    hold on
    maxc = zeros(1,Sensor_s.Nlambda);
    
    for iF = 1 : Sensor_s.Nlambda
        plot(theta, 10*log10(squeeze(abs(CBF(:,iF)))),'linewidth',2 ); hold on;
        leg{iF+1} = ['CBF ',num2str(f(iF)),' Hz']; 
        maxc(iF) = max(10*log10(squeeze(abs(CBF(:,iF)))));
    end
    
    plot(Signal_s.source_theta(1:end), -5, 'rx', 'Markersize', 20);
    lg = legend(leg); lg.FontSize=16;
    set(gca,'Fontsize',18)
    title('Single Freq. CBF');
    grid on;
    xlabel('Theta [deg.]'), ylabel('Power [dB]')
    xlim([thetalim(1) thetalim(2)]), ylim([-10 ceil(max(maxc))])
    %lg.Location = 'southwest';
    axP = get(gca,'Position');
    lg.Location = 'northeastoutside';
    set(gca, 'Position', axP)
    
    % Multi-F
    
    subplot(2,2,3);
    plot(Signal_s.source_theta(1), -5, 'rx', 'Markersize', 20);
    hold on
%   plot(theta,10*log10(squeeze(sum(abs(CBF),2))/Sensor_s.Nlambda), 'linewidth',2 ); % original
    plot(theta, squeeze(sum(10*log10(abs(CBF)),2)/Sensor_s.Nlambda),'linewidth',2 ); % modified by cfm 21.07.2020
    plot(Signal_s.source_theta(1:end), -5, 'rx', 'Markersize', 20);
    lg = legend('True source location','Multi-Freq. CBF');
    lg.FontSize=16;
    lg.Location = 'southwest';
    set(gca,'Fontsize',18)
    title('Incoherent Multi-Freq. CBF');
    grid on;
    xlabel('Theta [deg.]'), ylabel('Power [dB]')
    xlim([thetalim(1) thetalim(2)])
    ylim([-40 ceil(max(maxc))])
    
    
    % SBL
   % [gammak, Tind ]= maxk(gamma,15);
   % gammak = 10*log10(gammak);
    
    subplot(2,2,4);
    plot(Signal_s.source_theta(1), 0, 'rx', 'Markersize', 20); hold on;
    plot(theta, 10*log10(gamma),'linewidth',2);
    maxg = max(10*log10(gamma));
    %plot(theta(Tind), gammak, 'ko', 'Markersize', 20); hold on;
    plot(Signal_s.source_theta(1:end), 0, 'rx', 'Markersize', 20); hold on;
    lg=legend('True source location','SBL v.4');
    lg.Location = 'southwest';
    lg.FontSize=16;
    title('SBL');
    grid on; set(gca,'Fontsize',18)
    xlabel('Theta [deg.]'), ylabel('Power [dB]')
    xlim([thetalim(1) thetalim(2)])
    ylim([-50 maxg+2])
    
    
    
    %     subplot(2,2,4);
    %   %  plot(theta, abs(x_l1).^2); hold on;
    %     plot(Signal_s.source_theta, 0, 'rx', 'Markersize', 10);
    %     legend('X^2', 'True source location');
    %     title('L1 - CVX');
    %     grid on;
    
end

%end

function Signal_s = generate_signal(Sensor_s, Nsnapshot, snr, seed)
% function to generate sensor observations

% sensor settings
Nsensor = Sensor_s.Nsensor;
lambda = Sensor_s.lambda;
xq = Sensor_s.xq;

% three DOA example

source_theta = [-2; 3; 75]; source_amp = [0.5; 13; 1];

%source_theta = [-20 ]; source_amp = [1];

Nsources = length(source_theta)
Ysignal = zeros(Nsensor,Nsnapshot,Sensor_s.Nlambda);

% random number seed
rng(seed, 'twister');

for iF = 1 : Sensor_s.Nlambda
    
    % simulate source motion
    for t = 1:Nsnapshot
        Xsource = source_amp.*exp(1i*2*pi*rand(Nsources,1));    % random phase
        %Xsource=abs(Xsource);
        % Represenation matrix (steering matrix)
        u = sind(source_theta);
        A = exp(-1i*2*pi/lambda(iF)*xq*u.')/sqrt(Nsensor);
        
        % Signal without noise
        Ysignal(:,t,iF) = sum(A*diag(Xsource),2);
        
        % add noise to the signals
        rnl = 10^(-snr/20)*norm(Xsource);
        nwhite = complex(randn(Nsensor,1),randn(Nsensor,1))/sqrt(2*Nsensor);
        e = nwhite * rnl;	% error vector
        if ~(snr==100)
            Ysignal(:,t,iF) = Ysignal(:,t,iF) + e;
        end
    end
end
% output signal structure
Signal_s.Nsnapshot = Nsnapshot;
Signal_s.Nsources = Nsources;
Signal_s.source_theta = source_theta;
Signal_s.Xsource = Xsource;
Signal_s.Ysignal = Ysignal;

end
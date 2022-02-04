function [synthetic_signal, y_pos, rayleigh_pos_fit, frq_lut] = Synthetic_signal(width ,height, order, rayleigh, stokes, antistokes)
%height width inversion !

border = 8;
x_border = width * 0.12;
starting_frq = -7.5;
ending_frq = 7.5;

y_c = 0.5 ;
A = 500; %spacing between the orders (more or less)
b = (height/2) ^2 / (A - order*order) + 1; % b >= (y-yc)²/(A-n²) | 120 => impact y has on the whole => +/- curvature

A = 240000;
a = 1;
b = 0.4;
x_c = -300;
%y_c = 180;
min_x = 0 ;

min_x = sqrt(A - (order+1) * (order+1) - (0.5) * (0.5) / (b * b)) ;
max_x = sqrt(A - 2*2) ;
x_c = x_border / 2 ;
a = (width - x_border - x_c)/(max_x - min_x) ;
y_scale = 1 / (height ) ;



frq_lut = zeros(height, width, 'single');
signal = zeros(height, width, 'single');
signal_u16 = zeros(height, width, 'uint16');

rayleigh_pos = zeros(length(0+border:height-border), order, 'single'); % Initial
rayleigh_pos_remapped = zeros(length(0+border:height-border), order, 'single'); %Remapped by hand
rayleigh_pos_remapped_postfit = zeros(length(0+border:height-border), order, 'single'); %Remapped by function fitting
   
    function [y] = lorentzian(frq, amplitude, center, gamma)
        y = amplitude * gamma * gamma / (gamma * gamma + (frq - center)*(frq - center));
    end

    function [y] = root(x, a, b, c)
            delta =  b * b - 4 * a * (c - x );
            y = ((-b + sqrt(delta) )/ ( 2 * a)) ;
    end


first_line = border ;
for y = 0+border:height-border
    % === Create Rayleigh peaks
    for n=1:order
       x = abs(a) * (sqrt(A - (n+1) * (n+1) - (y * y_scale - y_c) * (y * y_scale - y_c) / (b * b)) - min_x) + x_c;
       rayleigh_pos(y - first_line + 1, order - n + 1) = x ;
       %frq_lut(y, round(x)) = 10 ;
    end
    
    % === Remap Rayleigh peaks
    step = (rayleigh_pos(y - first_line + 1, end) - rayleigh_pos(y - first_line + 1,1) ) / (order - 1);
    for n=1:order
        rayleigh_pos_remapped(y - first_line + 1, n) = rayleigh_pos(y - first_line + 1,1) + (n-1) * step ;
        x_new =  rayleigh_pos_remapped(y - first_line + 1, n);
        %frq_lut(y, round(x_new)) = 20 ;
    end
    
    
    %imagesc(frq_lut)
    %pause(0.1)
    
    p = polyfit(rayleigh_pos(y - first_line + 1, :), rayleigh_pos_remapped(y - first_line + 1, :), 2);
    rayleigh_pos_remapped_postfit(y - first_line + 1,:) = root(rayleigh_pos_remapped(y - first_line + 1,:),p(1), p(2), p(3));
    coeff(y - first_line + 1,:) = [p(1) p(2) p(3)] ;

    %Debug
%     hold on
%     plot(rayleigh_pos(y - first_line + 1, :), rayleigh_pos_remapped(y - first_line + 1, :));
%     plot(50:200, polyval(p, 50:200))
%     hold off
    
    % === create frq_lut    
    start_column = floor(rayleigh_pos(y - first_line + 1,1) - step / 2);  %Rough ROI
    end_column = ceil(rayleigh_pos(y - first_line + 1,order) + step / 2); % Rough ROI
    start_column = floor(root(rayleigh_pos_remapped(y - first_line + 1,2) - step / 2,p(1), p(2), p(3)));
    end_column = ceil(root(rayleigh_pos_remapped(y - first_line + 1,end) + step / 2,p(1), p(2), p(3)));
    for x=start_column:end_column
        x_remapped = polyval(p, x);
        dist = x_remapped - polyval(p, rayleigh_pos_remapped_postfit(y - first_line + 1,1))  ;
        pos = mod( dist, step);
        
        %Go to [-7.5 ; 7.5]
        frq = pos / step;
        if(frq > 0.5)
            frq = frq -1 ;
        end
        if(frq < -0.5)
            frq = frq + 1;
        end
        frq = frq + 0.5;
        frq = frq * (ending_frq - starting_frq) + starting_frq;

        frq_lut(y,x) = frq;
    end
    
    %Add functions    
    for x=start_column:end_column
       frq = frq_lut(y,x);
       signal(y,x) = signal(y,x) + lorentzian(frq, rayleigh(1), rayleigh(2), rayleigh(3)); % Rayleigh
       signal(y,x) = signal(y,x) + lorentzian(frq, stokes(1), stokes(2), stokes(3)); % Stokes
       signal(y,x) = signal(y,x) + lorentzian(frq, antistokes(1), antistokes(2), antistokes(3)); % antiStokes
    end
    
    signal_u16 = uint16( signal / max(max(signal)) * 65535 );
    
end
    
 %Display - for DEBUG
%  subplot(2,1,1)
%  hold on
%  imagesc(frq_lut)
%  for n = 1:order
%     plot(rayleigh_pos_remapped_postfit(:, n), 0+border:height-border);
%  end
%     axis([ 0 width 0 height]);
%  hold off
%  
%  subplot(2,1,2)
%  hold on
%  imagesc(signal_u16)
%   for n = 1:order
%     plot(rayleigh_pos_remapped_postfit(:, n), 0+border:height-border);
%   end
%     axis([ 0 width 0 height]);
%  hold off
 
 
 %return with correct format
 y_pos = (0+border:height-border)' ;
 synthetic_signal = signal_u16;
 rayleigh_pos_fit = rayleigh_pos_remapped_postfit(:,2:order); %First order is gibberish because the polynomial fit is convex
end
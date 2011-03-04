%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Joshua Mlcoch
%CS 4501 (computer vision)
%Assignment 1: Edge and corner detector
%1/14/2011
%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Read in source image
%%%%%%%%%%%%%%%%%%%%%%%%%%

source_image = imread('mandrill.jpg');

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%set the current colormap to gray
%%%%%%%%%%%%%%%%%%%%%%%%%%

colormap(gray);

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%convert to grayscale
%%%%%%%%%%%%%%%%%%%%%%%%%%

grayscale_image = rgb2gray(source_image);

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%convert to double
%%%%%%%%%%%%%%%%%%%%%%%%%%

double_image = im2double(grayscale_image);

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%apply the gaussian blur 
%%%%%%%%%%%%%%%%%%%%%%%%%%

[height, width] = size(double_image);  

sigma = 1;
T_h = 0.8;
T_l = 0.2;
kernel_size = 21; 

x_gauss = zeros(kernel_size, kernel_size);

y_gauss = zeros(kernel_size, kernel_size);

for i = 1:kernel_size
    
    for j = 1:kernel_size
        x_gauss(i, j) = -((j - ((kernel_size - 1) / 2) -1) / (sigma^2)) * exp ( - ((i - ((kernel_size - 1) / 2) -1)^2 + (j - ((kernel_size - 1) / 2) -1)^2) / (2 * sigma^2));
    end    
end

for i = 1:kernel_size
    
    for j = 1:kernel_size
        y_gauss(i, j) = -((i - ((kernel_size - 1) / 2) -1) / (sigma^2)) * exp ( - ((i - ((kernel_size - 1) / 2) -1)^2 + (j - ((kernel_size - 1) / 2) -1)^2) / (2 * sigma^2));
    end    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%compute the convolution
%%%%%%%%%%%%%%%%%%%%%%%%%%
dx = conv2(double_image, x_gauss);
dy = conv2(double_image, y_gauss);



%%%%%%%%%%%%%%%%%%%%%%%%%%
%%compute the gradient magintude
%%%%%%%%%%%%%%%%%%%%%%%%%%


grad =  zeros(height, width);



for i = 1 + ceil(kernel_size / 2):height - ceil(kernel_size / 2)  

    for j = 1 + ceil(kernel_size / 2):width - ceil(kernel_size / 2)  

        grad(i, j) = sqrt (dx(i, j)^2 + dy(i, j)^2 );

    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%apply non maximum suppression
%%%%%%%%%%%%%%%%%%%%%%%%%

image_non_max = grad; 

for i = 1 + ceil(kernel_size / 2):height - ceil(kernel_size / 2)  

    for j = 1 + ceil(kernel_size / 2):width - ceil(kernel_size / 2)  
        
        theta = atand(dx(i, j) / dy(i, j));
        
        if (theta < 0)
            theta = 360 + theta;
        end 
            
        if (0 < theta && theta <= 90)

            if(grad(i, j) < grad(i, j + 1) || grad(i, j) < grad(i, j - 1))

                image_non_max(i, j) = 0;

            end

        end
        
         if (90 < theta && theta <= 180)

            if(grad(i, j) < grad(i, j + 1) || grad(i, j) < grad(i, j - 1))

                image_non_max(i, j) = 0;

            end

         end
        
        
         if (180 < theta && theta <= 225)

            if(grad(i, j) < grad(i, j + 1) || grad(i, j) < grad(i, j - 1))

               image_non_max(i, j) = 0;

            end

         end
        
        if (225 < theta && theta <= 360)

            if(grad(i, j) < grad(i, j + 1) || grad(i, j) < grad(i, j - 1))

                image_non_max(i, j) = 0;

            end

        end
         
         
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%
%%apply post hysteresis
%%%%%%%%%%%%%%%%%%%%%%%%%


image_post_hy = image_non_max; 

for i = 1:height

    for j = 1:width 

        if (image_post_hy(i, j) >= T_h) 
            
            image_post_hy(i, j) = 2;

        end

    end

end

for i = 1:height

    for j = 1:width 

    
        
        if (image_post_hy(i, j) == 2) 
            
           k = i; 
           m = j;
            
           while (image_post_hy(k+1, m) > T_l)
               
               image_post_hy(k+1, m) = 2;
               k = k + 1;
               
           end 
           
           k = i; 
           m = j;
           
           while (image_post_hy(k+1, m+1) > T_l)
               
               image_post_hy(k+1, m+1) = 2;
               k = k + 1;
               m = m + 1;
               
           end 
           
           k = i; 
           m = j;
           
           while (image_post_hy(k-1, m-1) > T_l)
               
               image_post_hy(k-1, m-1) = 2;
               k = k - 1;
               m = m - 1;
               
           end 
           
           k = i; 
           m = j;
           
           while (image_post_hy(k-1, m) > T_l)
               
               image_post_hy(k-1, m) = 2;
               k = k + 1;
       
           end 
           
           k = i; 
           m = j;
           
           while (image_post_hy(k-1, m+1) > T_l)
               
               image_post_hy(k-1, m+1) = 2;
               k = k - 1;
               m = m + 1;
               
           end 
           
           k = i; 
           m = j;
           
           while (image_post_hy(k, m-1) > T_l)
               
               image_post_hy(k, m-1) = 2;
               m = m - 1;
               
           end 
           
           k = i; 
           m = j;
           
           while (image_post_hy(k, m+1) > T_l)
               
               image_post_hy(k, m+1) = 2;
               m = m + 1;
               
           end 
           
           k = i; 
           m = j;
           
           while (image_post_hy(k+1, m-1) > T_l)
               
               image_post_hy(k+1, m-1) = 2;
               k = k + 1;
               m = m - 1;
           end 
           
        end

    end

end
    

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Corner Detector
%%%%%%%%%%%%%%%%%%%%%%%%%


 k = 1;
 
 image_corner = grad;

 
for i = 2:height - 2

    for j = 2:width -2
        
    A = [image_corner(i-1, j-1) image_corner(i-1, j) image_corner(i-1, j+1); image_corner(i, j-1) image_corner(i, j) image_corner(i + 1, j + 1); image_corner(i + 1, j-1) image_corner(i + 1, j) image_corner(i + 1, j + 1)];
    
    A = cov(A);
    
    B = eig(A);
    
    small_l = min(B) * 1000000000000000000;
  
        if ( small_l > 30)
        
            list(:,:,:,k) = [i;j;small_l];
            
            k = k +1;
            
            double_image(i - 10, j -10) = 2;
             double_image(i - 10, j -9) = 2;
              double_image(i - 9, j -10) = 2;
               double_image(i - 9, j -9) = 2;
            
        end
    
    end 
    
end

imwrite(dx,'mandrill_dx.jpg');            

imwrite(dy,'mandrill_dy.jpg');            

imwrite(grad,'mandrill_gradient_mag.jpg');       

imwrite(image_non_max,'mandrill_non_max.jpg');

imwrite(image_post_hy,'mandrill_post_hy.jpg');

imwrite(double_image,'mandrill_corner.jpg'); 
 


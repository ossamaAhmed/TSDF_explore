close all
clear all
clc

results_folder = './trained_model/';
data_train = csvread([results_folder 'training_loss_results.csv']);
data_val = csvread([results_folder 'validation_loss_results.csv']);

figure; 
semilogy(data_train(:,1), data_train(:,2), 'b'); hold on; grid on;
semilogy(data_val(:,1), data_val(:,2), 'm');
%plot(data_train(:,1), data_train(:,2), 'b'); hold on; grid on;
%plot(data_val(:,1), data_val(:,2), 'm');

xlabel('Epoch', 'fontsize', 20);
ylabel('Mean L2 Loss', 'fontsize', 20);
title('Training Loss Curve', 'fontsize', 24);
legend('Training Data', 'Validation Data', 'location', 'best');

set(gcf, 'PaperPosition', [0 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [6 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'training_curve', 'pdf') %Save figure
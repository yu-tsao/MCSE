
addpath('D:\OneDrive\Main\Documents\Intern\2018SINICA\MatlabToolBox\P862_annex_A_2005_CD\source')
addpath('D:\OneDrive\Main\Documents\Intern\2018SINICA\MCME\UNet')
root_dir_1 = strcat('D:\OneDrive\Main\Documents\Intern\2018SINICA\MCME\UNet\Results_12356_set_20190214\','SincSDUN_FCN');
root_dir_2 = strcat('D:\OneDrive\Main\Documents\Intern\2018SINICA\MCME\UNet\Results_12356_set_20190214\','FCN');
root_dir_3 = strcat('D:\OneDrive\Main\Documents\Intern\2018SINICA\MCME\UNet\Results_12356_set_20190214\','rSDFCN');
root_dirs = {root_dir_1, root_dir_2, root_dir_3};
value_lists = cell(3,2);

set_list = {dir(root_dir_1), dir(root_dir_2), dir(root_dir_3)};
set_list{1} = set_list{1}(~ismember({set_list{1}.name},{'.','..','*xlsx','summary.xlsxsummary.xlsx','summary.xlsx'}));
set_list{2} = set_list{2}(~ismember({set_list{2}.name},{'.','..','*xlsx','summary.xlsxsummary.xlsx','summary.xlsx'}));
set_list{3} = set_list{3}(~ismember({set_list{3}.name},{'.','..','*xlsx','summary.xlsxsummary.xlsx','summary.xlsx'}));

for mm = 1:3
for j =2:2:4
    result_dir = fullfile(root_dirs{mm},set_list{mm}(j).name,'0718');%0718, 0326
    dir_list = dir(fullfile(result_dir,'*.wav'));
    values = zeros(length(dir_list)/2-20,3);
    for i = 1:size(values,1)
        clean = audioread(fullfile(result_dir, dir_list(2*i-1,1).name));
        predict = audioread(fullfile(result_dir, dir_list(2*i,1).name));
        clean = normalize_x(clean);
        predict = normalize_x(predict);
        s = stoi(clean,predict,16000);
        p = pesq_mex_vec(clean,predict,16000); 
        m = immse(clean,predict);
        values(i,:) = [s, p, m];
    end
    value_lists{mm,j/2} = values;
end
end

%%
figure; hold on;
for i=1:3
    plot(value_lists{i,1}(:,1))
end
legend('SincSDUN_FCN','FCN','rSDFCN')
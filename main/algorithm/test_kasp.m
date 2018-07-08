clear;
load('protein');
nlabel = max(gnd);
kasp_acc = zeros(10, 1);
cspec_acc =zeros(10, 1);
for p = 1:10
    m = p * 100;
    label = chen_kasp(fea, nlabel, m);
    label = bestMap(gnd, label);
    ac = sum(label == gnd) / size(gnd, 1);
    kasp_acc(p) = ac;
    fprintf('#landmarks = %d\n', m');
    fprintf('accuracy for kasp: %f\n', ac);
    label = chen_cspec(fea, nlabel, m);
    label = bestMap(gnd, label);
    ac = sum(label == gnd) / size(gnd, 1);
    cspec_acc(p) = ac;
    fprintf('accuracy for cspec: %f\n', ac);
end
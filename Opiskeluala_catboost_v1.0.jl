using Pkg, StatFiles, DataFrames, StatsBase, MLJ, StatisticalMeasures, CatBoost.MLJCatBoostInterface, Combinatorics, Shapley, HypothesisTests, AlgebraOfGraphics, CairoMakie, CSV
using Shapley: MonteCarlo
df = DataFrame(StatFiles.load("/home/matti/Documents/Huijarikysely.sav"))
# remove rows with missing values in the study subject column
df = dropmissing(df, :3)
numcases_tot = countmap(df[!, 3])

predictors = df[:,[:"question_1",:"question_2",:"question_4",:"question_8",:"question_9",:"question_10_row_1",:"question_10_row_2",:"question_10_row_3",:"question_10_row_4",:"question_10_row_5",:"question_10_row_6",:"question_10_row_7",:"question_11",:"question_12",:"question_13",:"question_14",:"question_15_row_1",:"question_15_row_2",:"question_15_row_3",:"question_15_row_4",:"question_15_row_5",:"question_15_row_6",:"question_15_row_7",:"question_15_row_8",:"question_15_row_9",:"question_15_row_10",:"question_15_row_11",:"question_15_row_12",:"question_15_row_13",:"question_15_row_14",:"question_16_row_1",:"question_16_row_2",:"question_16_row_3",:"question_16_row_4",:"question_16_row_5",:"question_16_row_6",:"question_16_row_7",:"question_16_row_8",:"question_16_row_9",:"question_16_row_10",:"question_16_row_11",:"question_16_row_12",:"question_16_row_13",:"question_16_row_14",:"question_16_row_15"]]

#impute missing values and set levels
imputer = FillImputer()
mach = machine(imputer, predictors) |> MLJ.fit!
predictors_imputed = MLJ.transform(mach, predictors)
#calculate together the number of matriculation examinations subjects as a new feature column
predictors_imputed[!, :question_10_sum] = combine(select(predictors_imputed, Cols(contains.("question_10"))), AsTable(:) .=> sum)[:,1]
#remove suspicious outlier datapoint with 35 matriculation examination subjects (!!)
outlier_ind = findall(predictors_imputed.question_10_sum .== 35)
deleteat!(predictors_imputed, outlier_ind)
deleteat!(df, outlier_ind)
#convert to strings
predictors_imputed = string.(predictors_imputed)
predictors_dict = [:question_1    => OrderedFactor, #Age
:question_2    => Multiclass, #Sex/Gender
:question_4    => OrderedFactor, #Start of studies in higher education (year)
:question_8    => Multiclass, #Previous degree, secondary education
:question_9    => OrderedFactor, #GPA limit of high school
:question_10_row_1    => OrderedFactor, #N grades in matriculation examination: L
:question_10_row_2    => OrderedFactor, #E
:question_10_row_3    => OrderedFactor, #M
:question_10_row_4    => OrderedFactor, #C
:question_10_row_5    => OrderedFactor, #B
:question_10_row_6    => OrderedFactor, #A
:question_10_row_7    => OrderedFactor, #I
:question_10_sum => OrderedFactor, #sum of all matriculation examination grades
:question_11    => Multiclass, #Mother's education
:question_12    => Multiclass, #Father's education
:question_13    => OrderedFactor, #Family's income level
:question_14    => OrderedFactor, #Family's societal class
:question_15_row_1    => OrderedFactor, #Family habits: "higher arts"
:question_15_row_2    => OrderedFactor, #newspapers
:question_15_row_3    => OrderedFactor, #discussion on societal matters
:question_15_row_4    => OrderedFactor, #academic / highly educated social environment
:question_15_row_5    => OrderedFactor, #economically / politically / culturally influential social environment
:question_15_row_6    => OrderedFactor, #reading aloud
:question_15_row_7    => OrderedFactor, #encouraged to study foreign languages
:question_15_row_8    => OrderedFactor, #encouraged in self-development
:question_15_row_9    => OrderedFactor, #encouraged to be goal-oriented
:question_15_row_10    => OrderedFactor, #encouraged to be curious
:question_15_row_11    => OrderedFactor, #encouraged to be independent
:question_15_row_12    => OrderedFactor, #encouraged to trust myself
:question_15_row_13    => OrderedFactor, #traveling abroad
:question_15_row_14    => OrderedFactor, #living abroad
:question_16_row_1    => OrderedFactor, #Family culture and attitudes: Mother's attitude towards academic performance
:question_16_row_2    => OrderedFactor, #Father's attitude towards academic performance
:question_16_row_3    => OrderedFactor, #Mother encouraged me to set my academic goals high 
:question_16_row_4    => OrderedFactor, #Father encouraged me to set my academic goals high
:question_16_row_5    => OrderedFactor, #My mother's view mattered in my educational choices
:question_16_row_6    => OrderedFactor, #My father's view mattered in my educational choices
:question_16_row_7    => OrderedFactor, #Attitude towards higher education
:question_16_row_8    => OrderedFactor, #Higher education was seen as a matter of course in my family
:question_16_row_9    => OrderedFactor, #Higher education was seen as a matter of course by myself
:question_16_row_10    => OrderedFactor, #Mother wanted me to get an academic degree
:question_16_row_11    => OrderedFactor, #Father wanted me to get an academic degree
:question_16_row_12    => OrderedFactor, #I was supported by advice from my family and relatives when applying to higher education
:question_16_row_13    => OrderedFactor, #I chose my field of study based on my family's professions / fields
:question_16_row_14    => OrderedFactor, #Mother believes in me
:question_16_row_15    => OrderedFactor] #Father believes in me
predictors_coerced = coerce(predictors_imputed, Dict(predictors_dict))
schema(predictors_coerced)

#predictor explanations
#1 #Ikä
#2 #Sukupuoli
#4 #Koulutuksen alkuvuosi
#8 #Toisen asteen tutkinto
#9 #Lukion keskiarvoraja
#10_1-7 #Ylioppilastodistuksen arvosanat (pitääkö summata?)
#11 #Äidin/huoltajan koulutus (numeroistaminen/standardisointi?)
#12 #Isän/huoltajan koulutus (numeroistaminen/standardisointi?)
#13 #Huoltajien toimeentulo
#14 #Lapsuudenperheen yhteiskuntaluokka
#15_1-14 #Lapsuudenperheen asenteet1
#16_1-15 #Lapsuudenperheen asenteet2

alat_dict = Dict(1.0  => "Eläinlääketieteellinen", 2.0 => "Liikuntatieteellinen", 3.0 => "Taideteollinen", 4.0 => "Farmasia", 5.0 => "Luonnontieteellinen", 6.0 => "Tanssi", 7.0 => "Hammaslääketieteellinen", 8.0 => "Lääketieteellinen", 9.0 => "Teatteri", 10.0 => "Humanistinen", 11.0 => "Maatalous-metsätieteellinen", 12.0 => "Teknistieteellinen", 13.0 => "Kasvatustieteellinen", 14.0 => "Musiikki", 15.0 => "Teologia", 16.0 => "Kauppatieteellinen", 17.0 => "Oikeustieteellinen", 18.0 => "Terveystieteellinen", 19.0 => "Kuvataide", 20.0 => "Psykologia", 21.0 => "Yhteiskuntatieteellinen")


#convert response variables to one-hot integers
responses = df[:,3]
onehot_responses = Int.(DataFrame(unique(responses) .== permutedims(responses), :auto))

#Main loop for model building and feature selection
collect_shapley_out = [];
collect_mean_shapley = [];
for response_var in 1:nrow(onehot_responses)
#Divide data and start with full model
    println("Building models for class \"", alat_dict[unique(responses)[response_var]], "\".")
    y = coerce(Vector{Int}(onehot_responses[response_var,:]), OrderedFactor);
    x = predictors_coerced;
    (Xtrain, Xtest), (ytrain, ytest) = partition((x, y), 0.7, rng=42, multi=true, shuffle=true);
    model = CatBoostClassifier(allow_writing_files=false, iterations = 1000);
    mach1 = machine(model, Xtrain, ytrain);
    #Evaluate with 5-fold CV
    try
        eval_metrics = evaluate!(mach1, resampling=CV(nfolds=5, rng=42), measure=auc);
        global mach1_auc = eval_metrics.measurement[1]
    catch
        println("CV not possible for class \"", alat_dict[unique(responses)[response_var]], "\" likely due to low number of cases.")
        global mach1_auc = 0.5
    end
    #Start variable selection if any predictive power in this class
    if mach1_auc > 0.55
        features = first.(sort(collect(mach1.report.vals[1].feature_importances), by = x -> x.second, rev = true));
        selections = Vector{Any}();
        for nrows in 1:length(features);
            rowvector = Vector{Symbol}();
            for feature in 1:nrows
                push!(rowvector, features[feature])
            end
            push!(selections, rowvector);
        end
        selections = vec.(selections);  
        compared_models = map(selections) do s
            FeatureSelector(features=s) |> model;
        end
        tmodel = TunedModel(models=compared_models, resampling=CV(nfolds=5, rng=42), measure=auc);
        mach2 = machine(tmodel, Xtrain, ytrain);
        MLJ.fit!(mach2)
        mach3 = machine(model, Xtrain[:,report(mach2).best_model.feature_selector.features], ytrain);
        eval_metrics2 = evaluate!(mach3, resampling=CV(nfolds=5, rng=42), measure=auc)
        #fit to full training data and calculate performance on test data
        MLJ.fit!(mach3)
        test_auc = auc(MLJ.predict(mach3, Xtest), ytest);
        #Calculate Shapley values in test data
        shapley_x = copy(Xtest[:,first.(mach3.report.vals[1].feature_importances)]);
        shapley_values = shapley(shapley_x -> MLJ.predict(mach3, shapley_x), MonteCarlo(CPUThreads(), 1024), shapley_x);
        #Format and gather output tables
        shapley_out = copy(shapley_x);
        for shapley_col in propertynames(shapley_values)
            shapley_out = hcat(shapley_out, DataFrame(pdf(shapley_values[shapley_col], [1]), [string(shapley_col, "_shapley")]));
        end
        mean_shapley_df = DataFrame(Feature = chop.(string.(propertynames(shapley_out)[ncol(shapley_x)+1:end]), head=0, tail=8), mean_effect = abs.(mean.(eachcol(shapley_out[:,ncol(shapley_x)+1:end]))));
        mean_shapley_df = sort(mean_shapley_df, order(:mean_effect, rev = true));
        mean_shapley_df.effect_scaled = mean_shapley_df.mean_effect/sum(mean_shapley_df.mean_effect);
        mean_shapley_df[!, :response_name] .= alat_dict[unique(responses)[response_var]];
        mean_shapley_df[!, :auc_test] .= test_auc;
        mean_shapley_df[!, :auc_train] .= eval_metrics2.measurement[1];
        mean_shapley_df[!, :auc_train_sem_x_1_96] .= sem(eval_metrics2.per_fold[1])*1.96;
        shapley_out[!, :response_name] .= alat_dict[unique(responses)[response_var]];
        shapley_out[!, :ytest] = ytest
        push!(collect_shapley_out, shapley_out);    
        push!(collect_mean_shapley, mean_shapley_df);
    end
end

shapley_df = vcat(collect_shapley_out..., cols=:union)
CSV.write("/home/matti/Documents/Huijarit/190625_shapley_df.csv", shapley_df)
shapley_df = CSV.read("/home/matti/Documents/Huijarit/190625_shapley_df.csv", DataFrame)

#Table 1
table1 = DataFrame(countmap(replace(responses, alat_dict ...)))
table1 = stack(table1, names(table1))
sort!(table1, [order(:value, rev=true)])
table1.percent = round.(table1[!, :value]/sum(table1[!, :value])*100, digits = 2)
push!(table1, ["Yhteensä", sum(table1.value), 100])
rename!(table1, [:Opiskeluala, :N, :%])
CSV.write("/home/matti/Documents/Huijarit/030725_Table1.csv", table1)
table1 = CSV.read("/home/matti/Documents/Huijarit/030725_Table1.csv", DataFrame)

#plotting
question_dict = Dict("question_1"    => "Ikä",#"Age",
"question_2"    => "Sukupuoli",#"Gender",
"question_4"    => "Aloitus koulutusohjelmassa (vuosi)",#"Start of studies in higher education (year)",
"question_8"    => "Toisen asteen tutkinto",#"Previous degree, secondary education",
"question_9"    => "Lukion keskiarvoraja",#"GPA limit of high school",
"question_10_row_1"    => "L arvosanoja ylioppilaskirjoituksissa",#"L grades in matriculation examination",
"question_10_row_2"    => "E arvosanoja ylioppilaskirjoituksissa",#"E grades in matriculation examination",
"question_10_row_3"    => "M arvosanoja ylioppilaskirjoituksissa",#"M grades in matriculation examination",
"question_10_row_4"    => "C arvosanoja ylioppilaskirjoituksissa",#"C grades in matriculation examination",
"question_10_row_5"    => "B arvosanoja ylioppilaskirjoituksissa",#"B grades in matriculation examination",
"question_10_row_6"    => "A arvosanoja ylioppilaskirjoituksissa",#"A grades in matriculation examination",
"question_10_row_7"    => "I arvosanoja ylioppilaskirjoituksissa",#"I grades in matriculation examination",
"question_10_sum" => "Ylioppilaskirjoitusten aineiden määrä",#"Sum of all matriculation examination grades",
"question_11"    => "Äidin koulutus",#"Mother's education",
"question_12"    => "Isän koulutus",#"Father's education",
"question_13"    => "Perheen toimeentulon taso",#"Family's income level",
"question_14"    => "Yhteiskuntaluokka",#"Family's societal class",
"question_15_row_1"    => "Korkeakulttuuri",#"Higher arts",
"question_15_row_2"    => "Sanomalehdet",#"Newspapers",
"question_15_row_3"    => "Keskustelu yhteiskunnallisista asioista",#"Discussion on societal matters",
"question_15_row_4"    => "Akateemisesti koulutettu kasvuympäristö",#"Academic social environment",
"question_15_row_5"    => "Vaikutusvaltainen kasvuympäristö",#"Societally influential social environment",
"question_15_row_6"    => "Ääneen lukeminen",#"Reading aloud",
"question_15_row_7"    => "Kannustus vieraiden kielten opiskeluun",#"Ecouraged to study foreign languages",
"question_15_row_8"    => "Kannustus itsensä kehittämiseen",#"Encouraged in self-development",
"question_15_row_9"    => "Kannustus korkeiden päämäärien asettamiseen",#"Encouraged to be goal-oriented",
"question_15_row_10"    => "Kannustus uteliaisuuteen",#"Encouraged to be curious",
"question_15_row_11"    => "Kannustus omatoimisuuteen",#"Encouraged to be independent",
"question_15_row_12"    => "Kannustus luottamaan itseensä",#"Encouraged to trust myself",
"question_15_row_13"    => "Ulkomaan matkailu",#"Traveling abroad",
"question_15_row_14"    => "Ulkomailla asuminen",#"Living abroad",
"question_16_row_1"    => "Äidin asenne koulumenestykseen",#"Mother's attitude towards academic performance",
"question_16_row_2"    => "Isän asenne koulumenestykseen",#"Father's attitude towards academic performance",
"question_16_row_3"    => "Äidin kannustus opiskelun korkeisiin tavoitteisiin",#"Mother encouraged me to set my academic goals high",
"question_16_row_4"    => "Isän kannustus opiskelun korkeisiin tavoitteisiin",#"Father encouraged me to set my academic goals high",
"question_16_row_5"    => "Äidin näkemyksien vaikutus koulutusvalintoihin",#"Mother's view mattered in my educational choices",
"question_16_row_6"    => "Isän näkemyksien vaikutus koulutusvalintoihin",#"Father's view mattered in my educational choices",
"question_16_row_7"    => "Yliopistokoulutuksen arvostus",#"Attitude towards higher education",
"question_16_row_8"    => "Yliopistokoulutuksen näkeminen itsestäänselvyytenä perheessä",#"Higher education was seen as a matter of course in my family",
"question_16_row_9"    => "Yliopistokoulutuksen näkeminen itsestäänselvyytenä itse",#"Higher education was seen as a matter of course by myself",
"question_16_row_10"    => "Äidin toive korkeakoulututkinnosta",#"Mother wanted me to get an academic degree",
"question_16_row_11"    => "Isän toive korkeakoulututkinnosta",#"Father wanted me to get an academic degree",
"question_16_row_12"    => "Perheen ja suvun neuvot koulutusvalintoja tehdessä",#"Advice from family when applying to higher education",
"question_16_row_13"    => "Perheen ja suvun alavalintojen vaikutus",#"Family's professions affected my choices",
"question_16_row_14"    => "Äidin usko menestykseen",#"Mother believes in me",
"question_16_row_15"    => "Isän usko menestykseen")#"Father believes in me")

mean_shapley_df = reduce(vcat, collect_mean_shapley)
shapley_df_levels = intersect(reverse(string.(first.(predictors_dict))), mean_shapley_df.Feature)
mean_shapley_df[!, :Feature_name] = replace(mean_shapley_df.Feature, question_dict...)
CSV.write("/home/matti/Documents/Huijarit/190625_mean_shapley_df.csv", mean_shapley_df)
mean_shapley_df = CSV.read("/home/matti/Documents/Huijarit/190625_mean_shapley_df.csv", DataFrame)

#Figure 1
figure_1_df = mean_shapley_df[:,[:response_name, :auc_test, :auc_train, :auc_train_sem_x_1_96]]
unique!(figure_1_df, "response_name")
#figure_1_df.sem_min = figure_1_df.auc_train - figure_1_df.auc_train_sem_x_1_96
#figure_1_df.sem_max = figure_1_df.auc_train + figure_1_df.auc_train_sem_x_1_96
figure_1_response_levels = sort(figure_1_df, order(:auc_test, rev = true)).response_name
figure_1_df = stack(figure_1_df, 2:3)
figure_1_df.auc_train_sem_x_1_96 = float.(@. ifelse(figure_1_df.variable == "auc_test", 0, figure_1_df.auc_train_sem_x_1_96))
figure_1_df.response_name = categorical(figure_1_df.response_name)
levels!(figure_1_df.response_name, figure_1_response_levels)

plt = data(figure_1_df) * (mapping(:response_name, :value, color = :variable, marker = :variable) * visual(Scatter; markersize = 15) + mapping(:response_name, :value, color = :variable, :auc_train_sem_x_1_96) * visual(Errorbars)) + (mapping([0.7]) * visual(HLines; linewidth = 3, color = (:red, 0.5), linestyle = :dash))
save("/home/matti/Documents/Huijarit/040725_Figure1.svg", AlgebraOfGraphics.draw(plt, scales(Marker = (; palette = [:xcross, :circle])), axis = (; xlabel = "Opiskelualamalli", ylabel = "AUC", backgroundcolor = :white, xticklabelrotation = 0.75), figure = (size = (800, 400), backgroundcolor = :white)))

filtered_palette = ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9"]
#Figure 2
mean_shapley_df_filtered = filter(:auc_test => >(0.7), mean_shapley_df)
mean_shapley_df_filtered.Feature = categorical(mean_shapley_df_filtered.Feature)
mean_shapley_df_filtered.Feature_name = categorical(mean_shapley_df_filtered.Feature_name)
feature_levels = [:"question_1",:"question_2",:"question_4",:"question_8",:"question_9",:"question_10_row_1",:"question_10_row_2",:"question_10_row_3",:"question_10_row_4",:"question_10_row_5",:"question_10_row_6",:"question_10_row_7","question_10_sum",:"question_11",:"question_12",:"question_13",:"question_14",:"question_15_row_1",:"question_15_row_2",:"question_15_row_3",:"question_15_row_4",:"question_15_row_5",:"question_15_row_6",:"question_15_row_7",:"question_15_row_8",:"question_15_row_9",:"question_15_row_10",:"question_15_row_11",:"question_15_row_12",:"question_15_row_13",:"question_15_row_14",:"question_16_row_1",:"question_16_row_2",:"question_16_row_3",:"question_16_row_4",:"question_16_row_5",:"question_16_row_6",:"question_16_row_7",:"question_16_row_8",:"question_16_row_9",:"question_16_row_10",:"question_16_row_11",:"question_16_row_12",:"question_16_row_13",:"question_16_row_14",:"question_16_row_15"]
feature_name_levels = replace(feature_levels, question_dict...)
levels!(mean_shapley_df_filtered.Feature, feature_levels)
levels!(mean_shapley_df_filtered.Feature_name, reverse(feature_name_levels))
axis = (width = 2000, height = 2000)
plt = data(mean_shapley_df_filtered) * mapping(:Feature_name, :mean_effect; color = :response_name => "Opiskeluala", dodge = :response_name) * visual(BarPlot, direction=:x, gap = 0.3)
save("/home/matti/Documents/Huijarit/040725_Figure2.svg", AlgebraOfGraphics.draw(plt, scales(Color = (; palette = filtered_palette)), axis = (; xlabel = "Normalisoitu Shapley arvo", ylabel = "", backgroundcolor = :white), figure = (size = (1000, 1500), backgroundcolor = :white)))

#print stats of the models
unique(mean_shapley_df[!, [:response_name, :auc_test]])
countmap(mean_shapley_df_filtered[!,4])
mean(collect(values(countmap(mean_shapley_df_filtered[!,4]))))

#plot Shapley values for comparisons between predictive models
plots_df = filter(row -> row[:response_name] in unique(mean_shapley_df_filtered.response_name), shapley_df)

#Figure 3
plots_df_3 = plots_df[:,[:response_name, :question_1, :question_1_shapley, :question_2, :question_2_shapley, :question_4, :question_4_shapley, :question_9, :question_9_shapley, :question_10_row_1, :question_10_row_1_shapley, :question_10_row_2, :question_10_row_2_shapley, :question_10_row_3, :question_10_row_3_shapley, :question_10_sum, :question_10_sum_shapley, :question_11, :question_11_shapley, :question_12, :question_12_shapley, :question_15_row_1, :question_15_row_1_shapley, :question_15_row_12, :question_15_row_12_shapley, :question_16_row_7, :question_16_row_7_shapley, :question_16_row_9, :question_16_row_9_shapley]]


plt = data(plots_df_3) * mapping(:question_1, :question_1_shapley, color = :response_name => "Opiskeluala", dodge_x = :response_name) * visual(RainClouds; clouds=nothing, plot_boxplots=false, markersize=10, jitter_width=0.13) + (mapping([0]) * visual(HLines; linewidth = 3, color = (:red, 0.5), linestyle = :dash))
save("/home/matti/Documents/Huijarit/040725_Figure3A.svg", AlgebraOfGraphics.draw(plt, scales(DodgeX = (; width = 0.8), Color = (; palette = filtered_palette)), axis = (; xticks = (1:6, ["19 v. tai alle", "20-24 v.", "25-29 v.", "30-34 v.", "35-39 v.", "40 v. tai yli"]), xlabel = "Ikä", ylabel = "Shapley arvo",  backgroundcolor = :white), figure = (size = (800, 400), backgroundcolor = :white)))

plt = data(dropmissing(plots_df_3, :question_10_row_1)) * mapping(:question_10_row_1, :question_10_row_1_shapley, color = :response_name => "Opiskeluala", dodge_x = :response_name) * visual(RainClouds; clouds=nothing, plot_boxplots=false, markersize=10, jitter_width=0.13) + (mapping([0]) * visual(HLines; linewidth = 3, color = (:red, 0.5), linestyle = :dash))
save("/home/matti/Documents/Huijarit/040725_Figure3B.svg", AlgebraOfGraphics.draw(plt, scales(DodgeX = (; width = 0.8), Color = (; palette = filtered_palette[Not(1)])), axis = (; xticks = 0:8, xlabel = "L arvosanoja ylioppilaskirjoituksissa", ylabel = "Shapley arvo",  backgroundcolor = :white), figure = (size = (800, 400), backgroundcolor = :white)))

plt = data(dropmissing(plots_df_3, :question_10_sum)) * mapping(:question_10_sum, :question_10_sum_shapley, color = :response_name => "Opiskeluala", dodge_x = :response_name) * visual(RainClouds; clouds=nothing, plot_boxplots=false, markersize=10, jitter_width=0.13) + (mapping([0]) * visual(HLines; linewidth = 3, color = (:red, 0.5), linestyle = :dash))
save("/home/matti/Documents/Huijarit/040725_Figure3C.svg", AlgebraOfGraphics.draw(plt, scales(DodgeX = (; width = 0.8), Color = (; palette = filtered_palette[Not(4)])), axis = (; xticks = 0:11, xlabel = "Ylioppilaskirjoitusten aineiden määrä", ylabel = "Shapley arvo",  backgroundcolor = :white), figure = (size = (1000, 400), backgroundcolor = :white)))

plt = data(dropmissing(plots_df_3, :question_11)) * mapping(:question_11, :question_11_shapley, color = :response_name => "Opiskeluala", dodge_x = :response_name) * visual(RainClouds; clouds=nothing, plot_boxplots=false, markersize=10, jitter_width=0.13) + (mapping([0]) * visual(HLines; linewidth = 3, color = (:red, 0.5), linestyle = :dash))
save("/home/matti/Documents/Huijarit/040725_Figure3D.svg", AlgebraOfGraphics.draw(plt, scales(DodgeX = (; width = 0.8), Color = (; palette = filtered_palette[Not(1)])), axis = (; xticks = (1:10, ["Peruskoulu tai kansakoulu", "Lukio", "Ammatillinen tutkinto", "Opistotutkinto", "Alempi korkeakoulututkinto yliopistossa", "Ylempi korkeakoulututkinto yliopistossa", "AMK-tutkinto", "Ylempi AMK-tutkinto", "Lisensiaatin tai tohtorin tutkinto", "En osaa sanoa/ Ei koske minua"]), xlabel = "Äidin koulutus", ylabel = "Shapley arvo",  backgroundcolor = :white, xticklabelrotation = 0.75), figure = (size = (1000, 600), backgroundcolor = :white)))

plt = data(dropmissing(plots_df_3, :question_12)) * mapping(:question_12, :question_12_shapley, color = :response_name => "Opiskeluala", dodge_x = :response_name) * visual(RainClouds; clouds=nothing, plot_boxplots=false, markersize=10, jitter_width=0.13) + (mapping([0]) * visual(HLines; linewidth = 3, color = (:red, 0.5), linestyle = :dash))
save("/home/matti/Documents/Huijarit/040725_Figure3E.svg", AlgebraOfGraphics.draw(plt, scales(DodgeX = (; width = 0.8), Color = (; palette = filtered_palette[Not(1)])), axis = (; xticks = (1:10, ["Peruskoulu tai kansakoulu", "Lukio", "Ammatillinen tutkinto", "Opistotutkinto", "Alempi korkeakoulututkinto yliopistossa", "Ylempi korkeakoulututkinto yliopistossa", "AMK-tutkinto", "Ylempi AMK-tutkinto", "Lisensiaatin tai tohtorin tutkinto", "En osaa sanoa/ Ei koske minua"]), xlabel = "Isän koulutus", ylabel = "Shapley arvo",  backgroundcolor = :white, xticklabelrotation = 0.75), figure = (size = (1000, 600), backgroundcolor = :white)))

plt = data(dropmissing(plots_df_3, :question_2)) * mapping(:question_2, :question_2_shapley, color = :response_name => "Opiskeluala", dodge_x = :response_name) * visual(RainClouds; clouds=nothing, plot_boxplots=false, markersize=10, jitter_width=0.13) + (mapping([0]) * visual(HLines; linewidth = 3, color = (:red, 0.5), linestyle = :dash))
save("/home/matti/Documents/Huijarit/040725_Figure4A.svg", AlgebraOfGraphics.draw(plt, scales(DodgeX = (; width = 0.8), Color = (; palette = filtered_palette[Not(1,4)])), axis = (; xticks = (1:4, ["Nainen", "Mies", "Muu", "En halua sanoa"]), xlabel = "Sukupuoli", ylabel = "Shapley arvo",  backgroundcolor = :white), figure = (size = (800, 400), backgroundcolor = :white)))

plt = data(dropmissing(plots_df_3, :question_4)) * mapping(:question_4, :question_4_shapley, color = :response_name => "Opiskeluala", dodge_x = :response_name) * visual(RainClouds; clouds=nothing, plot_boxplots=false, markersize=10, jitter_width=0.13) + (mapping([0]) * visual(HLines; linewidth = 3, color = (:red, 0.5), linestyle = :dash))
save("/home/matti/Documents/Huijarit/040725_Figure4B.svg", AlgebraOfGraphics.draw(plt, scales(DodgeX = (; width = 0.8), Color = (; palette = filtered_palette[Not(1,4)])), axis = (; xlabel = "Koulutuksen aloitusvuosi", ylabel = "Shapley arvo",  backgroundcolor = :white), figure = (size = (1000, 400), backgroundcolor = :white)))

plt = data(dropmissing(plots_df_3, :question_9)) * mapping(:question_9, :question_9_shapley, color = :response_name => "Opiskeluala", dodge_x = :response_name) * visual(RainClouds; clouds=nothing, plot_boxplots=false, markersize=10, jitter_width=0.13) + (mapping([0]) * visual(HLines; linewidth = 3, color = (:red, 0.5), linestyle = :dash))
save("/home/matti/Documents/Huijarit/040725_Figure4C.svg", AlgebraOfGraphics.draw(plt, scales(DodgeX = (; width = 0.8), Color = (; palette = filtered_palette[Not(1,4)])), axis = (; xticks = (1:6, ["9.0 tai korkeampi", "8.0-8.9", "7.0-7.9", "6.0-6.9", "alle 6.0", "En osaa sanoa"]), xlabel = "Lukion keskiarvoraja", ylabel = "Shapley arvo",  backgroundcolor = :white), figure = (size = (800, 400), backgroundcolor = :white)))

plt = data(dropmissing(plots_df_3, :question_10_row_2)) * mapping(:question_10_row_2, :question_10_row_2_shapley, color = :response_name => "Opiskeluala", dodge_x = :response_name) * visual(RainClouds; clouds=nothing, plot_boxplots=false, markersize=10, jitter_width=0.13) + (mapping([0]) * visual(HLines; linewidth = 3, color = (:red, 0.5), linestyle = :dash))
save("/home/matti/Documents/Huijarit/040725_Figure4D.svg", AlgebraOfGraphics.draw(plt, scales(DodgeX = (; width = 0.8), Color = (; palette = filtered_palette[Not(1,4)])), axis = (; xticks = 0:8, xlabel = "E arvosanoja ylioppilaskirjoituksissa", ylabel = "Shapley arvo",  backgroundcolor = :white), figure = (size = (800, 400), backgroundcolor = :white)))

plt = data(dropmissing(plots_df_3, :question_10_row_3)) * mapping(:question_10_row_3, :question_10_row_3_shapley, color = :response_name => "Opiskeluala", dodge_x = :response_name) * visual(RainClouds; clouds=nothing, plot_boxplots=false, markersize=10, jitter_width=0.13) + (mapping([0]) * visual(HLines; linewidth = 3, color = (:red, 0.5), linestyle = :dash))
save("/home/matti/Documents/Huijarit/040725_Figure4E.svg", AlgebraOfGraphics.draw(plt, scales(DodgeX = (; width = 0.8), Color = (; palette = filtered_palette[Not(1,4)])), axis = (; xticks = 0:8, xlabel = "M arvosanoja ylioppilaskirjoituksissa", ylabel = "Shapley arvo",  backgroundcolor = :white), figure = (size = (800, 400), backgroundcolor = :white)))

plt = data(dropmissing(plots_df_3, :question_15_row_1)) * mapping(:question_15_row_1, :question_15_row_1_shapley, color = :response_name => "Opiskeluala", dodge_x = :response_name) * visual(RainClouds; clouds=nothing, plot_boxplots=false, markersize=10, jitter_width=0.13) + (mapping([0]) * visual(HLines; linewidth = 3, color = (:red, 0.5), linestyle = :dash))
save("/home/matti/Documents/Huijarit/040725_Figure4F.svg", AlgebraOfGraphics.draw(plt, scales(DodgeX = (; width = 0.8), Color = (; palette = filtered_palette[Not(1,3)])), axis = (; xticks = (0:4, ["Tyhjä", "Ei lainkaan", "Vähän", "Jonkin verran", "Paljon"]), xlabel = "Korkeakulttuuri", ylabel = "Shapley arvo",  backgroundcolor = :white), figure = (size = (800, 400), backgroundcolor = :white)))

plt = data(dropmissing(plots_df_3, :question_15_row_12)) * mapping(:question_15_row_12, :question_15_row_12_shapley, color = :response_name => "Opiskeluala", dodge_x = :response_name) * visual(RainClouds; clouds=nothing, plot_boxplots=false, markersize=10, jitter_width=0.13) + (mapping([0]) * visual(HLines; linewidth = 3, color = (:red, 0.5), linestyle = :dash))
save("/home/matti/Documents/Huijarit/040725_Figure4G.svg", AlgebraOfGraphics.draw(plt, scales(DodgeX = (; width = 0.8), Color = (; palette = filtered_palette[Not(1,4)])), axis = (; xticks = (0:4, ["Tyhjä", "Ei lainkaan", "Vähän", "Jonkin verran", "Paljon"]), xlabel = "Kannustus luottamaan itseensä", ylabel = "Shapley arvo",  backgroundcolor = :white), figure = (size = (800, 400), backgroundcolor = :white)))

plt = data(dropmissing(plots_df_3, :question_16_row_7)) * mapping(:question_16_row_7, :question_16_row_7_shapley, color = :response_name => "Opiskeluala", dodge_x = :response_name) * visual(RainClouds; clouds=nothing, plot_boxplots=false, markersize=10, jitter_width=0.13) + (mapping([0]) * visual(HLines; linewidth = 3, color = (:red, 0.5), linestyle = :dash))
save("/home/matti/Documents/Huijarit/040725_Figure4H.svg", AlgebraOfGraphics.draw(plt, scales(DodgeX = (; width = 0.8), Color = (; palette = filtered_palette[Not(1,2)])), axis = (; xticks = (0:5, ["Tyhjä", "EOS", "Täysin eri mieltä", "Jossain määrin eri mieltä", "Jossain määrin samaa mieltä", "Täysin samaa mieltä"]), xlabel = "\"Lapsuudenperheessäni arvostettiin yliopistokoulutusta\"", ylabel = "Shapley arvo",  backgroundcolor = :white, xticklabelrotation = 0.75), figure = (size = (800, 600), backgroundcolor = :white)))

plt = data(dropmissing(plots_df_3, :question_16_row_9)) * mapping(:question_16_row_9, :question_16_row_9_shapley, color = :response_name => "Opiskeluala", dodge_x = :response_name) * visual(RainClouds; clouds=nothing, plot_boxplots=false, markersize=10, jitter_width=0.13) + (mapping([0]) * visual(HLines; linewidth = 3, color = (:red, 0.5), linestyle = :dash))
save("/home/matti/Documents/Huijarit/040725_Figure4I.svg", AlgebraOfGraphics.draw(plt, scales(DodgeX = (; width = 0.8), Color = (; palette = filtered_palette[Not(3,4)])), axis = (; xticks = (0:5, ["Tyhjä", "EOS", "Täysin eri mieltä", "Jossain määrin eri mieltä", "Jossain määrin samaa mieltä", "Täysin samaa mieltä"]), xlabel = "\"Akateemisen koulutuksen hankkiminen on ollut minulle nuoruudesta lähtien itsestäänselvä asia\"", ylabel = "Shapley arvo",  backgroundcolor = :white, xticklabelrotation = 0.75), figure = (size = (800, 600), backgroundcolor = :white)))

#Poimintoja yksittäisistä piirteistä:
#lääketieteellinen yhteiskuntaluokka
med_class = dropmissing(plots_df[:, [:response_name, :question_14, :question_14_shapley]])
AlgebraOfGraphics.draw(data(med_class) * mapping(:question_14, :question_14_shapley) * visual(RainClouds; clouds=nothing, plot_boxplots=false, markersize=10, jitter_width=0.13) + (mapping([0]) * visual(HLines; linewidth = 3, color = (:red, 0.5), linestyle = :dash)), axis = (; xticks = (1:6, ["Yläluokka", "Ylempi keskiluokka", "Alempi keskiluokka", "Työväenluokka", "Jokin muu", "En osaa sanoa"]), xlabel = "Yhteiskuntauokka", ylabel = "Shapley arvo", title = "(Lääketieteellinen malli)",  backgroundcolor = :white), figure = (size = (800, 400), backgroundcolor = :white))
#lääketieteellinen ääneen lukeminen
med_reading = dropmissing(plots_df[:, [:response_name, :question_15_row_6, :question_15_row_6_shapley]])
combine(groupby(med_reading, :question_15_row_6), [:question_15_row_6_shapley] .=> mean)
AlgebraOfGraphics.draw(data(med_reading) * mapping(:question_15_row_6, :question_15_row_6_shapley) * visual(RainClouds; clouds=nothing, plot_boxplots=false, markersize=10, jitter_width=0.13) + (mapping([0]) * visual(HLines; linewidth = 3, color = (:red, 0.5), linestyle = :dash)), axis = (; xticks = (0:4, ["Tyhjä", "Ei lainkaan", "Vähän", "Jonkin verran", "Paljon"]), xlabel = "\"Minulle luettiin ääneen\"", ylabel = "Shapley arvo", title = "(Lääketieteellinen malli)",  backgroundcolor = :white), figure = (size = (800, 400), backgroundcolor = :white))
#keskustelu yhteiskunnallisista asioista
politics = dropmissing(plots_df[:, [:response_name, :question_15_row_3, :question_15_row_3_shapley]])
AlgebraOfGraphics.draw(data(politics) * mapping(:question_15_row_3, :question_15_row_3_shapley, color = :response_name => "Opiskeluala", dodge_x = :response_name) * visual(RainClouds; clouds=nothing, plot_boxplots=false, markersize=10, jitter_width=0.13) + (mapping([0]) * visual(HLines; linewidth = 3, color = (:red, 0.5), linestyle = :dash)), scales(DodgeX = (; width = 0.8), Color = (; palette = filtered_palette[Not(1,2,4)])), axis = (; xticks = (0:4, ["Tyhjä", "Ei lainkaan", "Vähän", "Jonkin verran", "Paljon"]), xlabel = "\"Keskustelimme ajankohtaisista yhteiskunnallisista asioista\"", ylabel = "Shapley arvo",  backgroundcolor = :white), figure = (size = (800, 400), backgroundcolor = :white))
#kannustus uteliaisuuteen
curiosity = dropmissing(plots_df[:, [:response_name, :question_15_row_10, :question_15_row_10_shapley]])
AlgebraOfGraphics.draw(data(curiosity) * mapping(:question_15_row_10, :question_15_row_10_shapley, color = :response_name => "Opiskeluala", dodge_x = :response_name) * visual(RainClouds; clouds=nothing, plot_boxplots=false, markersize=10, jitter_width=0.13) + (mapping([0]) * visual(HLines; linewidth = 3, color = (:red, 0.5), linestyle = :dash)), scales(DodgeX = (; width = 0.8), Color = (; palette = filtered_palette[Not(1,3,5)])), axis = (; xticks = (0:4, ["Tyhjä", "Ei lainkaan", "Vähän", "Jonkin verran", "Paljon"]), xlabel = "\"Minua kannustettiin uteliaisuuteen\"", ylabel = "Shapley arvo",  backgroundcolor = :white), figure = (size = (800, 400), backgroundcolor = :white))

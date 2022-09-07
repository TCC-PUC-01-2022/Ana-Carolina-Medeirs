import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

dados = pd.read_csv('Presos.csv')



variaveis = ['Sex','Age','SpanishLatinoHispanic','BlackNonBlack','USBorn','MaritalStatus_NeverMarried','MaritalStatu_Separated','MaritalStatus_Married','MaritalStatus_Divorced','MaritalStatus_Widowed'
,'USMilitaryService','CurrentlySentencedToServeTime','AgeAtFirstArrest','BeforeAdmissionHighestGradeOfSchoolAttended_HighSchool','BeforeAdmissionHighestGradeOfSchoolAttended_College'
,'BeforeAdmissionHighestGradeOfSchoolAttended_Elementary','BeforeAdmissionHighestGradeOfSchoolAttended_GraduateSchool','BeforeAdmissionHighestGradeOfSchoolAttended_NeverAttended'
,'CompletedThatYear','DuringTheMonthBeforeArrestHadJobOrBusiness','BeforeArrestEverHomeless','AnyFamilyEverBeenSentencedAndServedTimeInJailOrPrison','WhileGrowingUpHaveFriendsUsingDrugs'
,'WhileGrowingUpHaveFriendsDestroyingOrDamagingProperty','WhileGrowingUpHaveFriendsShoplifting','WhileGrowingUpHaveFriendsStealingMotorVehiclesOrPartsFromMotorVehicles','WhileGrowingUpHaveFriendsSellingStolenProperty'
,'WhileGrowingUpHaveFriendsBreakingIntoHomesOrOtherBuildings','WhileGrowingUpHaveFriendsSellingImportingOrManufacturingDrug','WhileGrowingUpHaveFriendssMuggingRobbingOrExtorting'
,'WhileGrowingUpHaveFriendsAnyOtherIllegalActivity','BeforeAdmissionAnyoneEverPressuredOrForcedYouToHaveAnySexualContact','BeforeAdmissonEverBeenPhysicallyAbused','AgeWhenFirstStartedDrinking'
,'EverDrunkAlcoholicBeveragesMoreThanOnceAWeekForMoreThanAMonth','InYearBeforeOffenseDidYouDrinkAnyAlcohol','DrinkingAnyAlcoholAtTimeOffense','EverUsedHeroin','EverUsedOtherOpiate','EverUsedMethamphetamine'
,'EverUsedOtherAmphetamine','EverUsedMethaqualone','EverUsedBarbiturates','EverUsedTranquilizers','EverUsedCrack','EverUsedCocaine','EverUsedPcp','EverUsedEcstasy','EverUsedLsd','EverUsedMarijuanaOrHashish',
'EverUsedAnyOtherDrugsThatWeDidnTMention','EverInhaledOrSniffedSubstancesToGetHigh','AgeFirstTimeYouUsedAnyOfTheseDrugs','EverAttendedAnyKindOfAlcoholOrDrugTreatmentProgram','AtAdmissionTakingPrescribedMedication'
,'TestResultTb_Negative','TestResultTb_Positive','TestResultTb_ResultNotAvailableYet','SinceAdmissionHadMedicalExamination','SinceAdmissionBloodTest','SinceAdmissionIntentionallyInjured',
'EverDiagnosedADepressiveDisorder','EverDiagnosedManicDepressionBipolarDisorderOrMania','EverDiagnosedSchizophreniaOrAnotherPsychoticDisorder','EverDiagnosedPostTraumaticStressDisorder'
,'EverDiagnosedAnotherAnxietyDisorderSuchAsAPanicDisorder','EverDiagnosedAPersonalityDisorder','EverDiagnosedAnyOtherMentalOrEmotionalCondition','EverReceivedCounselingFromTrainedProfessional',
'EverReceivedOtherMentalHealthTreatmentOrServices','EverAttemptedSuicide','EverConsideredSuicide','DifficultySeeingNewsprintEvenWithGlasses','DifficultyHearingConversationEvenWithHearingAid'
,'LearningDisability','SpeechImpairment','UseAidsToHelpWithDailyActivities','EnrolledInSpecialEducationClasses','ConsiderYourselfToHaveADisability','HoursSpentWhereYouSleepLast24Hour'
,'SpentTimeInPhysicalExerciseLast24Hours','TelevisionAvailableInThisPrison','WatchAnyTelevisionLast24Hours','NewspapersMagazinesOrBooksAvailable','SpentAnyTimeReadingLast24Hours',
'SpendAnyTimeInOtherRecreationLast24Hours','EngagedInAnyReligiousActivities','TelephoneFriendsAndFamily','NmbrOfTelephoneCallsHaveYouMadeOrReceivedLastWeek','VisitsLastMonth','WorkAssignmentOffPrisonGrounds'
,'WorkAssignment','SinceAdmissionAnyVocationalOrJobTrainingProgram','SinceAdmissionAnyOtherEducationProgram','ParticipatedInReligiousStudyGroupes','ParticipatedInAnEthnicOrRacialOrganization',
'ParticipatedInInmateAssistanceGroups','ParticipatedInOtherInmateSelfHelpGroups','ParticipatedInEmploymentCounseling','ParticipatedInClassesInParentingOrChildRearingSkills',
'ParticipatedInClassesInLifeSkillsAndCommunityAdjustment','ParticipatedInOtherPreReleasePrograms','GivenFurloughOrDayPass','TypeOfOffenses_Violent','TypeOfOffenses_Drug','TypeOfOffenses_Property'
,'TypeOfOffenses_PublicOrder','PrevProbation','UsedAlcohol','UsedDrugs','VeteranStatus','GedEarned','AnyChildrenIncludingStepOrAdoptedChildren','SinceAdmissionHowOftenHaveYouMadeOrReceivedCallsFromChildren_Monthly'
,'SinceAdmissionHowOftenHaveYouMadeOrReceivedCallsFromChildren_Never','SinceAdmissionHowOftenHaveYouMadeOrReceivedCallsFromChildren_Weekly','SentOrReceivedMailFromChildren_Monthly',
'SentOrReceivedMailFromChildren_Weekly','SentOrReceivedMailFromChildren_Never','VisitedByChildren_Weekly','VisitedByChildren_Never','VisitedByChildren_Monthly','HivTested','ResultOfLastTestHIV_Negative'
,'ResultOfLastTestHIV_ResultNotAvailableYet','ResultOfLastTestHIV_Positive','AllowedToHaveVisits','PrevIncarceration','ParticipatedInAnyOfTheseActivities']
#print (dados.head())

x = dados[variaveis]
y = dados['WrittenUpOrFoundGuiltyOfBreakingAnyRules']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

modelo = SVC(kernel='linear')
#modelo = SVC(C=1.0, kernel='poly', gamma='auto')

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print('Acurácia:', accuracy_score(y_test, y_pred) * 100)
print('F-Measure:', f1_score(y_test, y_pred, pos_label=1, average=None) * 100)
print('Precisão:', precision_score(y_test, y_pred, pos_label=1, average=None) * 100)
print('Recall:', recall_score(y_test, y_pred, pos_label=1, average=None) * 100)
package fakhraei_kdd2015;

import org.linqs.psl.application.inference.MPEInference
import org.linqs.psl.application.inference.result.FullInferenceResult;
import org.linqs.psl.application.learning.weight.em.HardEM
import org.linqs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE
import org.linqs.psl.config.*
import org.linqs.psl.core.*
import org.linqs.psl.core.inference.*
import org.linqs.psl.database.DataStore
import org.linqs.psl.database.Database
import org.linqs.psl.database.DatabasePopulator
import org.linqs.psl.database.Partition
import org.linqs.psl.database.rdbms.RDBMSDataStore
import org.linqs.psl.database.rdbms.driver.H2DatabaseDriver
import org.linqs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import org.linqs.psl.groovy.*
import org.linqs.psl.model.atom.QueryAtom
import org.linqs.psl.model.predicate.Predicate
import org.linqs.psl.model.term.ConstantType;
import org.linqs.psl.model.atom.*
import org.linqs.psl.ui.loading.*
import org.linqs.psl.util.database.*
import org.linqs.psl.utils.dataloading.InserterUtils;
import org.linqs.psl.utils.evaluation.result.*
import org.linqs.psl.utils.evaluation.statistics.RankingScore
import org.linqs.psl.utils.evaluation.statistics.SimpleRankingComparator

// Checking the arguments
if (args.length!=5){
	System.out.println "\nUsage: SpamModel [subModel:1,2,3] [totalFolds] [testFold] [validationFold] [dataFolder]";
	System.out.println "Example: SpamModel 1 3 3 2 'data/'";
	System.exit(0)
}
int model = args[0].toInteger()
int numberOfFolds = args[1].toInteger()
int testFold = args[2].toInteger()
int validationFold = args[3].toInteger()
def base_dir = args[4];

System.out.println "\nStarting..."
System.out.println("totalFolds: "+numberOfFolds)
System.out.println("testFold: "+testFold)
System.out.println("validationFold: "+validationFold)
System.out.println("dataFolder: "+base_dir)

// Setting the config file parameters
ConfigManager cm = ConfigManager.getManager();
ConfigBundle bundle_cfg = cm.getBundle("fakhraei_kdd2015");

// Settings the experiments parameters
today = new Date();
double initialWeight = 1;
boolean sq = true;



// Setting up the database
String dbpath = "./psl_db"+today.getDate()+""+today.getHours()+""+today.getMinutes()+""+today.getSeconds();
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Memory, dbpath, true), bundle_cfg);

// Creating the PSL Model
// ======================
PSLModel m = new PSLModel(this,data)

// Defining Predicates
m.add predicate: "spammer",	types: [ConstantType.UniqueID]
m.add predicate: "prior_credible",	types: [ConstantType.UniqueID]
m.add predicate: "credible",	types: [ConstantType.UniqueID]
m.add predicate: "report",	types: [ConstantType.UniqueID , ConstantType.UniqueID]

// Adding rules
if (model == 1)
	{
	// Model 1
	m.add rule : report(User2,User1) >> spammer(User1),  weight : initialWeight, squared: sq
	m.add rule : ~spammer(User) ,  weight : initialWeight, squared: sq
	}
else if (model == 2)
	{
	// Model 2
	m.add rule : (credible(User2) & report(User2,User1)) >> spammer(User1),  weight : initialWeight, squared: sq
	m.add rule : prior_credible(User2) >> credible(User2),  weight : initialWeight, squared: sq
	m.add rule : ~prior_credible(User2) >> ~credible(User2),  weight : initialWeight, squared: sq
	m.add rule : ~spammer(User) ,  weight : initialWeight, squared: sq
	}
else if (model == 3)
	{
	// Model 3
	m.add rule : (credible(User2) & report(User2,User1)) >> spammer(User1),  weight : initialWeight, squared: sq
	m.add rule : (spammer(User1) & report(User2,User1)) >> credible(User2),  weight : initialWeight, squared: sq
	m.add rule : (~spammer(User1) & report(User2,User1)) >> ~credible(User2),  weight : initialWeight, squared: sq
	m.add rule : prior_credible(User2) >> credible(User2),  weight : initialWeight, squared: sq
	m.add rule : ~prior_credible(User2) >> ~credible(User2),  weight : initialWeight, squared: sq
	m.add rule : ~spammer(User) ,  weight : initialWeight, squared: sq
	}

// Printing the model
System.out.println m;

// Creating the partition to read the data
Partition read_pt = data.getPartition("read");
Partition write_pt = data.getPartition("write");
Partition labels_pt = data.getPartition("labels");

Partition read_wl_pt = data.getPartition("wl_read");
Partition write_wl_pt = data.getPartition("wl_write");
Partition labels_wl_pt = data.getPartition("wl_labels");

// Reading from file
	System.out.println "Loading spammer ...";

	//Loading the train set
	for (int i=1; i<=numberOfFolds; i++)
	{
		if (i!=testFold){
			insert = data.getInserter(spammer, read_pt)
			InserterUtils.loadDelimitedDataTruth(insert, base_dir+'spammer_fold_'+i+'.tsv')			
		}
	}	

	//loding the test labels
	insert = data.getInserter(spammer, labels_pt)
	InserterUtils.loadDelimitedDataTruth(insert, base_dir+'spammer_fold_'+testFold+'.tsv')
	
	//Loading the test set
	insert = data.getInserter(spammer, write_pt)
	InserterUtils.loadDelimitedData(insert, base_dir+'spammer_fold_'+testFold+'_nolabel.tsv')


	// Loadining the train set for WeightLearning
	for (int i=1; i<=numberOfFolds; i++)
	{
		if ((i!=testFold)&&(i!=validationFold)){
			insert = data.getInserter(spammer, read_wl_pt)
			InserterUtils.loadDelimitedDataTruth(insert, base_dir+'spammer_fold_'+i+'.tsv')
		}
	}	

	// Loading the validation set labels
	insert = data.getInserter(spammer, labels_wl_pt)
	InserterUtils.loadDelimitedDataTruth(insert, base_dir+'spammer_fold_'+validationFold+'.tsv')

	// Loading the validation set 
	insert = data.getInserter(spammer, write_wl_pt)
	InserterUtils.loadDelimitedData(insert, base_dir+'spammer_fold_'+testFold+'_nolabel.tsv')
	InserterUtils.loadDelimitedData(insert, base_dir+'spammer_fold_'+validationFold+'_nolabel.tsv')


	System.out.println "Loading prior_credibility ...";

	insert = data.getInserter(prior_credible, read_pt)
	InserterUtils.loadDelimitedDataTruth(insert, base_dir+'prior_credibility_test_fold_'+testFold+'.tsv')

	insert = data.getInserter(credible, write_pt)
	InserterUtils.loadDelimitedDataTruth(insert, base_dir+'prior_credibility_test_fold_'+testFold+'.tsv')


	insert = data.getInserter(prior_credible, read_wl_pt)
	InserterUtils.loadDelimitedDataTruth(insert, base_dir+'prior_credibility_weightlearning_fold_'+testFold+'.tsv')

	insert = data.getInserter(credible, write_wl_pt)
	InserterUtils.loadDelimitedDataTruth(insert, base_dir+'prior_credibility_weightlearning_fold_'+testFold+'.tsv')

		
	System.out.println "Loading reported ...";
	
	insert = data.getInserter(report, read_pt)
	InserterUtils.loadDelimitedData(insert, base_dir+'reported.tsv')
	
	insert = data.getInserter(report, read_wl_pt)
	InserterUtils.loadDelimitedData(insert, base_dir+'reported.tsv')

// Setting which predicates are closed
Set <Predicate>closedPredicates = [prior_credible, report];


// Weight Learning
timeNow = new Date();
System.out.println("\nWeight Learning Start: "+timeNow);
System.out.println("-------------------\n");

Database wl_train_db = data.getDatabase(write_wl_pt, closedPredicates, read_wl_pt);
Database wl_labels_db = data.getDatabase(labels_wl_pt, [spammer] as Set);

HardEM wLearn = new HardEM(m, wl_train_db, wl_labels_db, bundle_cfg);
wLearn.learn();

wl_train_db.close();
wl_labels_db.close();

System.out.println m;


// Inference
timeNow = new Date();
System.out.println("Infernece Start: "+timeNow);
System.out.println("-------------------\n");

Database inference_db = data.getDatabase(write_pt, closedPredicates ,read_pt);

MPEInference mpe = new MPEInference(m, inference_db, bundle_cfg);
FullInferenceResult result = mpe.mpeInference();
mpe.close();
mpe.finalize();

inference_db.close();

timeNow = new Date();

System.out.println("End: "+timeNow);
System.out.println("-------------------\n");


System.out.println("Evaluting ...");

def labels_db = data.getDatabase(labels_pt, closedPredicates)
Database predictions_db = data.getDatabase(data.getPartition("100"), write_pt)

def comparator = new SimpleRankingComparator(predictions_db)
comparator.setBaseline(labels_db)

// Choosing what metrics to report
def metrics = [RankingScore.AUPRC, RankingScore.NegAUPRC,  RankingScore.AreaROC]
double [] score = new double[metrics.size()]

try {
	for (int i = 0; i < metrics.size(); i++) {
		comparator.setRankingScore(metrics.get(i))
		score[i] = comparator.compare(spammer)
	}

	System.out.println("\nArea under positive-class PR curve: " + score[0])
	System.out.println("Area under negative-class PR curve: " + score[1])
	System.out.println("Area under ROC curve: " + score[2])
}
catch (ArrayIndexOutOfBoundsException e) {
	System.out.println("No evaluation data! Terminating!");
}

package lazic;

import java.nio.file.Path;

import lazic.sources.*;
import lazic.sources.GlobalAquacultureProduction;
import lazic.sources.NzRoadFatalities;
import lazic.utils.ingest.CsvLongParser;
import lazic.utils.ingest.IngestManager;

public class Main {
	public static void main(String[] args) {
		// Register data sources

		new NzBusinessConfidence();
		new NzGdp();
		new NzRatesFx();
		new NzVehicleRegistrations();
		new YfFinances();
		new YfPrices();
		new NzLaborStats();
 		new NzRoadFatalities();
		new NzLaborTaxation();
		new NzPensions();
		new NzTaxRevenue();
		new NzBalanceOfPayments();
		new WikimediaPageviews();
		new GlobalFoodPrices();
		 new GlobalAquacultureProduction(); 

		IngestManager.INSTANCE.fetchDataFromSources();
		IngestManager.INSTANCE.printSubset(100);

		Path dataDir = Path.of("").toAbsolutePath().resolve("data");
		dataDir.toFile().mkdirs();
		String out = dataDir.resolve("data_long.csv").toString();

		CsvLongParser.saveCsv(out);
	}
}
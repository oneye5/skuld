package lazic;

import java.nio.file.Path;

import lazic.sources.*;
import lazic.sources.GlobalAquacultureProduction;
import lazic.sources.NzRoadFatalities;
import lazic.utils.RunLog;
import lazic.utils.ingest.CsvLongParser;
import lazic.utils.ingest.IngestManager;

public class Main {
	public static void main(String[] args) {
		Path dataDir = Path.of("").toAbsolutePath().resolve("data");
		dataDir.toFile().mkdirs();

		// Install runtime-error capture before any work so source ctors,
		// uncaught exceptions, and stderr output are all logged to disk.
		RunLog.install(dataDir.resolve("ingest_errors.log"));

		// Register data sources

		new NzBusinessConfidence();
		new NzGdp();
		new NzRatesFx();
		new NzVehicleRegistrations();
		new YfFinances();
		new YfSector();
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

		String out = dataDir.resolve("data_long.csv").toString();

		CsvLongParser.saveCsv(out);
	}
}
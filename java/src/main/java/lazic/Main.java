package lazic;

import java.nio.file.Path;

import lazic.sources.*;
import lazic.sources.GlobalAquacultureProduction;
import lazic.sources.examples.NzRoadFatalities;
import lazic.utils.ingest.CsvLongParser;
import lazic.utils.ingest.IngestManager;

public class Main {
	public static void main(String[] args) {
		new NzBusinessConfidence();
		new NzGdp();
		new NzRatesFx();
		new NzVehicleRegistrations();
		new YfFinances();
		new YfPrices();
		new NzLaborStats();
 		new NzRoadFatalities();
		new GlobalAquacultureProduction();
		IngestManager.INSTANCE.fetchDataFromSources();
		IngestManager.INSTANCE.printSubset(100);

		String out = Path.of("")
						.toAbsolutePath()
						.getParent()
						.toString()
						+ "\\data\\data_long.csv";

		CsvLongParser.saveCsv(out);
	}
}
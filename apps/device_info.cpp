#include <cuda_runtime_api.h>
#include <cstdlib>
#include <cstdio>

#include <QtCore/QDateTime>
#include <QtCore/QString>
#include <QtCore/QStringList>
#include <QtCore/QChar>
#include <QtCore/QVariant>
#include <QtCore/QSettings>

int main(int /*argc*/, char** /*argv*/)
{
    size_t free, total;
    cudaError_t error = cudaMemGetInfo(&free, &total);
    int device_id;
    cudaGetDevice(&device_id);
    printf("Device [%i] Error code = %i, Memory: free = %.3f MB, total = %.3f MB.\n",
           device_id, error, free/(1024.0*1024.0), total/(1024.0*1024.0));

//    printf("\n\n");
//    QString date_time_string = "10-12-2011 20:30:02.999";
//    QDateTime date_time = QDateTime::fromString(date_time_string, "dd-MM-yyyy hh:mm:ss.zzz");
//    QTime time = date_time.time();
//    QDate date = date_time.date();
//    printf("day    = %i\n", date.day());
//    printf("month  = %i\n", date.month());
//    printf("year   = %i\n", date.year());
//    printf("hour   = %i\n", time.hour());
//    printf("min    = %i\n", time.minute());
//    printf("sec    = %i\n", time.second());
//    printf("ms     = %i\n\n", time.msec());
//
//    QString lon_string = "W 51 45 07.30";
//    QString lat_string = "N 01 15 28.15";
//
//    QStringList lon = lon_string.split(" ");
//    printf("long: %s %02i %02i %2.3f\n", lon.at(0).toLatin1().data(), lon.at(1).toInt(),
//            lon.at(2).toInt(), lon.at(3).toDouble());
//
//    QStringList lat = lat_string.split(" ");
//    printf("lat: %s %02i %02i %2.3f\n", lat.at(0).toLatin1().data(), lat.at(1).toInt(),
//            lat.at(2).toInt(), lat.at(3).toDouble());
//
//    QString ra_string  = "23:12:26.02";
//    QStringList ra = ra_string.split(":");
//    printf("RA: %02i %02i %2.3f\n", ra.at(0).toInt(), ra.at(1).toInt(),
//            ra.at(2).toDouble());
//
//
//    QString dec_string = "58 48 43.04";
//    QStringList dec = dec_string.split(" ");
//    printf("Dec: %02i %02i %2.3f\n", dec.at(0).toInt(), dec.at(1).toInt(),
//            dec.at(2).toDouble());
//
//    QString string_list = "2000,3000,4000";
//    QStringList int_list = string_list.split(",");
//    QList<int> int_values;
//    for (int i = 0; i < int_list.size(); ++i)
//    {
//        int_values[i] = int_list[i].toInt();
//        printf("%i\n", int_values[i]);
//    }
//
//    QVariant test_variant = string_list;
//    printf("type = %s\n", test_variant.typeName());
//
//    QSettings s;
//    s.setValue("date", date_time);
//
//    QVariant v = s.value("date");
//    printf("type = %s\n", v.typeName());

    return EXIT_SUCCESS;
}




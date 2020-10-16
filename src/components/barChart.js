import React from "react";
import { Bar } from "react-chartjs-2";
import { MDBContainer } from "mdbreact";

class BarChart extends React.Component {
    state = {
      dataBar: {
        labels: ["Youtube", "Reddit", "Hacker News"],
        datasets: [
          {
            label: "",
            data: this.props.resultingValues,
            backgroundColor: [
              "rgba(255, 134,159,0.4)",
              "rgba(98,  182, 239,0.4)",
              "rgba(255, 218, 128,0.4)",
            ],
            borderWidth: 2,
            borderColor: [
              "rgba(255, 134, 159, 1)",
              "rgba(98,  182, 239, 1)",
              "rgba(255, 218, 128, 1)",
            ]
          }
        ]
      },
      barChartOptions: {
        legend: {
          display: false
        },
        responsive: true,
        maintainAspectRatio: true,
        scales: {
          xAxes: [
            {
              barPercentage: 1,
              gridLines: {
                display: true,
                color: "rgba(0, 0, 0, 0.1)"
              }
            }
          ],
          yAxes: [
            {
              gridLines: {
                display: true,
                color: "rgba(0, 0, 0, 0.1)"
              },
              ticks: {
                beginAtZero: true
              }
            }
          ]
        }
      }
    }
  
  
    render() {
      return (
        <MDBContainer>
          <h3 className="mt-5">Where does your comment match?</h3>
          <Bar data={this.state.dataBar} options={this.state.barChartOptions} />
        </MDBContainer>
      );
    }
  }
  
  export default BarChart;
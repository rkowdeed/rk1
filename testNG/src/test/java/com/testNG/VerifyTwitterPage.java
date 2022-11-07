package com.testNG;

import static org.junit.Assert.assertEquals;

import java.util.concurrent.TimeUnit;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;  
import org.openqa.selenium.chrome.ChromeDriver;  
import org.testng.annotations.AfterTest;  
import org.testng.annotations.BeforeTest;  
import org.testng.annotations.Test;  
public class VerifyTwitterPage {
public String baseUrl = "https://google.com/";
String driverPath = "src//test//resources//chromedriver.exe";  
public WebDriver driver ;   
/*
 * Steps being performed in the Test.
 * 1. Launch chrome browser and open twitter website (https://twitter.com/)
 * 2. Click on “Terms of service” link (Present under sign in options)
 * 3. Switch to the new page and verify “Terms of service” page is displayed
 * 4.  Close the new window and switch back to main twitter page
 * 5.  Identify the “Sign up with Google” button using X-path and perform click operation.
 * 6. Verify the new page.
 * 7. Close window & quit browser
 */
@Test             
public void test() {      
// set the system property for Chrome driver      
System.setProperty("webdriver.chrome.driver", driverPath);  
// Create driver object for CHROME browser  
driver = new ChromeDriver();  
driver.manage().timeouts().implicitlyWait(20, TimeUnit.SECONDS);  
driver.manage().window().maximize();  
//1. Launch chrome browser and open twitter website (https://twitter.com/)
driver.get(baseUrl);  

driver.manage().timeouts().implicitlyWait(20, TimeUnit.SECONDS);
//2. Click on “Terms of service” link (Present under sign in options)
driver.findElement(By.xpath("//a[text()='Terms of service']")).click();
driver.manage().timeouts().implicitlyWait(20, TimeUnit.SECONDS);
//3. Switch to the new page and verify “Terms of service” page is displayed
String twitterMainPageWindowHandle = driver.getWindowHandle();
for(String winHandle : driver.getWindowHandles()){
    driver.switchTo().window(winHandle);
}

assertEquals("The Twitter Terms Of Service Page is not opened", driver.getTitle(), "Twitter Terms Of Service");
driver.manage().timeouts().implicitlyWait(20, TimeUnit.SECONDS);
//4.  Close the new window and switch back to main twitter page
driver.close();
driver.switchTo().window(twitterMainPageWindowHandle);
Runtime.getRuntime().exit(0);
//5.  Identify the “Sign up with Google” button using X-path and perform click operation.
//Sign Up With Google Link - Using Xpath
//driver.manage().timeouts().implicitlyWait(20, TimeUnit.SECONDS);
//driver.findElement(By.xpath("//*[@id=\"container\"]/div/div[2]/span[1]")).click();

//6. Verify the new page.
//assertEquals("The Sign Up With Google Link not opened", driver.getTitle(), "Sign In - Google Accounts");
//driver.manage().timeouts().implicitlyWait(20, TimeUnit.SECONDS);

}     
@BeforeTest  
public void beforeTest() {    
/*
 * Perform any Test Data Setup if required here.  
 */
}     
@AfterTest  
public void afterTest() {  

//7. Close window & quit browser	
driver.quit();  
}         
}

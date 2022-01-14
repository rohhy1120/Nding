/* 이름: demo_madang.sql */
/* 설명 */
 
/* root 계정으로 접속, madang 데이터베이스 생성, madang 계정 생성 */
/* MySQL Workbench에서 초기화면에서 +를 눌러 root connection을 만들어 접속한다. */
DROP DATABASE IF EXISTS  dancearch;
DROP USER IF EXISTS  dancearch@localhost;
create user dancearch@localhost identified WITH mysql_native_password  by 'dancearch';
create database dancearch;
grant all privileges on dancearch.* to dancearch@localhost with grant option;
commit;

/* madang DB 자료 생성 */
/* 이후 실습은 MySQL Workbench에서 초기화면에서 +를 눌러 madang connection을 만들어 접속하여 사용한다. */
 
USE dancearch;

CREATE TABLE Song (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `title` varchar(30) NOT NULL,
  `singer` varchar(30) NOT NULL,
  `album` varchar(30) NOT NULL,
  `link`     varchar(100) NOT NULL,
   PRIMARY KEY (`id`)
);

INSERT INTO Song VALUES (1,'가시나','선미','img','www.youtube.com');
INSERT INTO Song VALUES (2,'강남스타일','싸이','img','www.youtube.com');
INSERT INTO Song VALUES (3,'나야나','워너원','img','www.youtube.com');
INSERT INTO Song VALUES (4,'뉴페이스','싸이','img','www.youtube.com');
INSERT INTO Song VALUES (5,'러브인','싸이','img','www.youtube.com');
INSERT INTO Song VALUES (6,'문을여시오','임창정','img','www.youtube.com');
INSERT INTO Song VALUES (7,'백팩키드','백팩키드','img','www.youtube.com');
INSERT INTO Song VALUES (8,'불장난','블랙핑크','img','www.youtube.com');
INSERT INTO Song VALUES (9,'빨간맛','현아','img','www.youtube.com');
INSERT INTO Song VALUES (10,'뿜뿜','모모랜드','img','www.youtube.com');
INSERT INTO Song VALUES (11,'아무노래','지코','img','www.youtube.com');
INSERT INTO Song VALUES (12,'위아래','EXID','img','www.youtube.com');
INSERT INTO Song VALUES (13,'Hip','마마무','img','www.youtube.com');
INSERT INTO Song VALUES (14,'Not today','방탄소년단','img','www.youtube.com');
INSERT INTO Song VALUES (15,'Pick Me','IOI','img','www.youtube.com');
INSERT INTO Song VALUES (16,'RolyPoly','티아라','img','www.youtube.com');
INSERT INTO Song VALUES (17,'Snapping','청하','img','www.youtube.com');
INSERT INTO Song VALUES (18,'TT','트와이스','img','www.youtube.com');
INSERT INTO Song VALUES (19,'yesoryes','트와이스','img','www.youtube.com');

SELECT * FROM Song WHERE id = 1

commit;